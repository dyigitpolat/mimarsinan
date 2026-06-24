"""Residual-block (``y = x + F(x)``) torch-NF == deployed-HCM-sim fidelity lock.

Proves the residual-connection design (``docs/research/RESIDUAL_MAPPING_DESIGN.md``):
a residual ADD lowers to a host-side param-free ``ComputeAdapter(operator.add)``
ComputeOp whose ``input_sources`` span BOTH predecessors (the diamond — one
producer fanning to the skip and the F branch), and deploys bit-exact through the
same NF↔SCM↔HCM packing the existing fidelity harness locks for any ComputeOp.

Two fixtures, two granularity claims:

- ``_OnChipSkipResidual`` — the skip producer is an **on-chip** neural core, so the
  add is a **cross-segment** host merge (a sync-point, like the harness's LayerNorm
  ``_SyncPoint``). Each branch is decoded to a rate (``/T``) before the host add, so
  the merge runs purely in the rate domain. This is **bit-exact across every
  bit-exact mode** (``lif``/``ttfs_cycle_based``/``ttfs_quantized``) × every lossless
  config (``identity``/``neuron_split``/``axon_fuse``), exactly as the design's
  forward-semantics section specifies.

- ``DeepMLP(residual=True)`` — the registered depth-probe model with a skip across a
  hidden block. With ``nn.Flatten`` the stem subsumes into the host encoding layer,
  so the skip source is a host value and the add is an **in-segment** ComputeOp. This
  is bit-exact for **LIF** (the lossless-capable mode, the design's "ship now"
  claim). In-segment TTFS value-domain merge is NOT bit-exact by construction (the
  branches are not decoded to rates first) and is the documented Tier-0→Tier-1 seam,
  not a regression of this change — no production forward code is added; the add
  routes through the pre-existing ``ComputeAdapter`` path.

No existing parity test is weakened; this file only adds residual coverage.
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.nn as nn

from integration._torch_sim_fidelity import (
    assert_config_triggered,
    assert_torch_sim_fidelity,
    build_torch_and_hcm,
    mapping_configs,
    mapping_structure,
)
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ShapeMismatchError
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.deep_mlp import DeepMLP
from mimarsinan.torch_mapping.converter import convert_torch_model

T = 8
INPUT_SHAPE = (16,)
NUM_CLASSES = 10
WIDTH = 24
BIT_EXACT_MODES = ["lif", "ttfs_cycle_based", "ttfs_quantized"]
LOSSLESS_CONFIGS = ["identity", "neuron_split", "axon_fuse"]

# Widths 24 (> 8 split budget) and fan-in 24+bias (> 16 fuse budget) so
# neuron_split tiles and axon_fuse fuses on the residual model's on-chip layers.
_CFGS = mapping_configs(wide_dim=64, split_neurons=8, fuse_core_axons=16)


class _OnChipSkipResidual(nn.Module):
    """``z = relu(stem(x)); y = z + relu(F(z)); head(y)`` with the SKIP source on-chip.

    No ``Flatten``: the stem stays an on-chip neural core, so its output ``z`` is a
    real spike-domain segment output. The add therefore becomes a cross-segment
    rate-domain merge (a sync-point) — the topology the design's bit-exact-across-
    modes claim requires. ``F`` is a single equal-width layer (the design's minimal
    ``y = z + Linear(z)``), so the add is a bare elementwise add (no projection).

    F-depth matters for ``ttfs_quantized``: a single-layer F merges bit-exact across
    every mode, but a deeper F-branch before the cross-segment merge leaves a
    quantization residual in ``ttfs_quantized`` (the value-domain re-encode does not
    compose losslessly through multiple TTFS layers feeding one merge). The minimal
    residual sidesteps that; the deeper-F TTFS residual is a documented seam.
    """

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.stem = nn.Linear(16, width)
        self.a0 = nn.ReLU()
        self.f1 = nn.Linear(width, width)
        self.af1 = nn.ReLU()
        self.head = nn.Linear(width, NUM_CLASSES)
        self.ah = nn.ReLU()

    def forward(self, x):
        z = self.a0(self.stem(x))
        b = self.af1(self.f1(z))
        y = z + b
        return self.ah(self.head(y))


def _samples(n, seed):
    torch.manual_seed(seed)
    return torch.rand(n, 16)


def _ir_for(model):
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024)
    ir.map(repr_)
    return ir


def _add_compute_ops(ir):
    return [
        n for n in ir.nodes
        if isinstance(n, ComputeOp)
        and isinstance(n.params.get("module"), ComputeAdapter)
        and n.params["module"].fn is operator.add
    ]


# --- Test A: IR representation — the residual add is a param-free diamond merge ---

class TestResidualAddIsADiamondComputeOp:
    def test_on_chip_skip_emits_one_param_free_add_spanning_both_branches(self):
        """Exactly one ``ComputeAdapter(operator.add)`` ComputeOp, ``_bound_count==0``,
        ``input_sources`` spanning two DISTINCT predecessor node_ids (the diamond)."""
        ir = _ir_for(_OnChipSkipResidual())
        adds = _add_compute_ops(ir)
        assert len(adds) == 1, f"expected exactly one residual add ComputeOp, got {len(adds)}"
        add = adds[0]
        assert add.params["module"]._bound_count == 0, (
            "a residual add is param-free (no bound tensors) — stays picklable"
        )
        sources = add.input_sources.flatten()
        predecessor_ids = sorted({int(s.node_id) for s in sources})
        assert len(predecessor_ids) == 2, (
            f"the residual add must span two distinct predecessors (skip + F), "
            f"got node_ids={predecessor_ids} — not a diamond"
        )
        # Equal-width bare add: both branches are 'width' wide → 2*width sources.
        assert len(sources) == 2 * WIDTH, (
            f"equal-width bare add should concat 2*width sources, got {len(sources)}"
        )

    def test_deep_mlp_residual_flag_emits_the_add(self):
        """``DeepMLP(residual=True)`` routes its skip through the same generic add."""
        model = DeepMLP(
            input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
            depth=4, width=WIDTH, residual=True,
        )
        adds = _add_compute_ops(_ir_for(model))
        assert len(adds) >= 1, "DeepMLP(residual=True) must emit at least one residual add"
        for add in adds:
            assert add.params["module"]._bound_count == 0


# --- Test B: bit-exact forward — torch NF == deployed sim across modes/configs ---

class TestResidualBitExactForward:
    @pytest.mark.parametrize("mode", BIT_EXACT_MODES)
    @pytest.mark.parametrize("config_name", LOSSLESS_CONFIGS)
    def test_on_chip_skip_residual_bit_exact_all_modes(self, mode, config_name):
        """Cross-segment (sync-point) residual merge: float64 ``atol=0`` (LIF also
        per-neuron ``k==k``) for every bit-exact mode × every lossless config."""
        torch.manual_seed(0)
        flow, hcm, hybrid, nodes = build_torch_and_hcm(
            _OnChipSkipResidual(), INPUT_SHAPE, NUM_CLASSES,
            spiking_mode=mode, config=_CFGS[config_name], T=T,
        )
        # The residual on-chip skip makes this a 2-segment (sync-point) model.
        assert mapping_structure(hybrid)["neural_segments"] == 2, (
            "on-chip skip residual must split into 2 neural segments (sync-point merge)"
        )
        assert_config_triggered(hybrid, config_name)
        result = assert_torch_sim_fidelity(
            flow, hcm, hybrid, nodes, _samples(16, 11),
            spiking_mode=mode, config_name=config_name, T=T,
        )
        assert result["granularity"] == "bit_exact"
        assert result["out_max_abs"] == 0.0
        if mode == "lif":
            assert result.get("per_neuron_perceptrons", 0) > 0

    @pytest.mark.parametrize("config_name", LOSSLESS_CONFIGS)
    def test_deep_mlp_residual_flag_lif_bit_exact(self, config_name):
        """``DeepMLP(residual=True)`` (in-segment host add) is LIF bit-exact across
        every lossless config — the lossless-capable 'ship now' mode."""
        model = DeepMLP(
            input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
            depth=4, width=WIDTH, residual=True,
        )
        torch.manual_seed(0)
        flow, hcm, hybrid, nodes = build_torch_and_hcm(
            model, INPUT_SHAPE, NUM_CLASSES,
            spiking_mode="lif", config=_CFGS[config_name], T=T,
        )
        assert_config_triggered(hybrid, config_name)
        result = assert_torch_sim_fidelity(
            flow, hcm, hybrid, nodes, _samples(16, 11),
            spiking_mode="lif", config_name=config_name, T=T,
        )
        assert result["granularity"] == "bit_exact"
        assert result["out_max_abs"] == 0.0
        assert result.get("per_neuron_perceptrons", 0) > 0


# --- Test C: a width-mismatched bare add is rejected (projection-skip requirement) ---

class TestWidthMismatchedBareAddRejected:
    """A dimension-changing residual is NOT a bare add — it needs a projection core
    on the skip branch first. Two rejection seams cover the two authoring paths."""

    def test_fx_traced_unequal_width_add_is_rejected_at_conversion(self):
        """An FX-traced bare add of unequal widths fails at ``convert_torch_model``:
        torch's own ShapeProp broadcast check fires during tracing (surfaced as a
        ``TracingError``) before mapping is even reached."""
        from mimarsinan.torch_mapping.torch_graph_tracer import TracingError

        class _BadResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Linear(16, 24)
                self.a0 = nn.ReLU()
                self.proj = nn.Linear(24, 12)  # F branch narrows to 12
                self.ap = nn.ReLU()
                self.head = nn.Linear(12, NUM_CLASSES)

            def forward(self, x):
                z = self.a0(self.stem(x))       # width 24
                b = self.ap(self.proj(z))       # width 12 — mismatched
                return self.head(z + b)         # bare add of 24 + 12

        with pytest.raises((TracingError, ShapeMismatchError, RuntimeError)):
            convert_torch_model(_BadResidual(), INPUT_SHAPE, NUM_CLASSES, device="cpu")

    def test_programmatic_mapper_unequal_width_add_raises_shape_mismatch(self):
        """The programmatic ComputeOpMapper path (the SkipPerceptronMixer authoring
        style) raises the design's exact ``ShapeMismatchError`` from
        ``_check_broadcastable`` when the two branches do not broadcast — the seam
        the design names. Driven directly so the check is exercised, not torch's."""
        from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

        mapper = ComputeOpMapper(
            [object(), object()],  # two sources → multi-input add path
            ComputeAdapter(operator.add),
        )
        with pytest.raises(ShapeMismatchError):
            mapper._check_broadcastable((torch.zeros(1, 24), torch.zeros(1, 12)))
