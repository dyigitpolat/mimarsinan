"""D2 — Residual Tier-1 as a VALID, characterized on-chip deployment.

Tier-0 adds the two residual streams **host-side** (``ComputeAdapter(operator.add)``);
it is bit-exact but breaks pure-on-chip deployment (a host/chip round-trip).
Tier-1 (``IRMapping(onchip_residual_merge=True)``, default OFF) lowers a param-free
equal-width add onto the crossbar as a signed-IF identity-merge core, so the residual
stays on-chip in ONE neural segment.

Wave-7 D2A established the decomposition (``residual_tier1_intrinsic_limit.md``):

  * Component A — a shared-HCM-fill latency-window alignment (closeable, lives outside
    this unit's files); after it, NF == HCM at float64 atol=0.
  * Component B — INTRINSIC: the in-segment IF head re-quantizes the merged spike
    train differently than the host-add re-encode, by **exactly 1 spike (~1/T)** by
    construction. Re-uniformizing on-chip needs a host round-trip == Tier 0.

So Tier-1 is NOT bit-exact to Tier-0; it is a different, VALID deployment whose
difference is bounded by the characterized ~1/T re-quant. This file MEASURES that
bounded effect on a small residual model and locks:

  1. DEFAULT OFF is byte-identical to the Tier-0 host-add deployment.
  2. The flag-on Tier-1 merge DEPLOYS (a real packed HCM, single on-chip segment).
  3. The deployed Tier-1-vs-Tier-0 delta is BOUNDED by the ~1/T re-quant (<= 1 spike
     per output on the closeable-A floor; <= 2 spikes including the unfixed
     Component-A alignment), NEVER an unbounded blow-up.

These are real measurements through the production config gate (``IRMapping`` ->
packed HCM), not a harness back door.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest
import torch
import torch.nn as nn

from integration._torch_sim_fidelity import mapping_configs, mapping_structure
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8
INPUT_SHAPE = (16,)
NUM_CLASSES = 10
WIDTH = 24
_CFGS = mapping_configs(wide_dim=64, split_neurons=8, fuse_core_axons=16)


class _OnChipSkipResidual(nn.Module):
    """``z = relu(stem(x)); y = z + relu(F(z)); head(y)`` with the SKIP source on-chip.

    No ``Flatten``: the stem stays an on-chip neural core, so its output ``z`` is a
    real spike-domain segment output and the add is a bare equal-width residual.
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


def _build_lif_hcm(model, *, onchip_residual_merge, config_name="identity", t_steps=T):
    """Convert ``model`` and pack it into a deployed LIF HCM through the production
    config gate. Returns ``(hcm, hybrid)``. ``onchip_residual_merge`` selects the
    Tier-1 on-chip merge (default off keeps the Tier-0 host add)."""
    config = _CFGS[config_name]
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=t_steps, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=config.ir_max_axons, max_neurons=config.ir_max_neurons,
        allow_coalescing=config.allow_coalescing,
        onchip_residual_merge=onchip_residual_merge,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{
            "max_axons": config.core_max_axons,
            "max_neurons": config.core_max_neurons,
            "count": 4000,
        }],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=config.allow_neuron_splitting,
            allow_coalescing=config.allow_coalescing,
        ),
    )
    hcm = SpikingHybridCoreFlow(
        INPUT_SHAPE, hybrid, simulation_length=t_steps, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return hcm, hybrid


def _build_lif_hcm_no_flag(model, config_name="identity"):
    """Build a deployed LIF HCM constructing ``IRMapping`` WITHOUT the new flag
    argument at all — the literal pre-Tier-1 (Tier-0) call. Proves the new
    argument's default is byte-identical to the old signature."""
    config = _CFGS[config_name]
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=config.ir_max_axons, max_neurons=config.ir_max_neurons,
        allow_coalescing=config.allow_coalescing,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{
            "max_axons": config.core_max_axons,
            "max_neurons": config.core_max_neurons,
            "count": 4000,
        }],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=config.allow_neuron_splitting,
            allow_coalescing=config.allow_coalescing,
        ),
    )
    return SpikingHybridCoreFlow(
        INPUT_SHAPE, hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    ), hybrid


def _add_compute_ops(ir):
    return [
        n for n in ir.nodes
        if isinstance(n, ComputeOp)
        and isinstance(n.params.get("module"), ComputeAdapter)
        and n.params["module"].fn is operator.add
    ]


def _ir_for(model, *, onchip_residual_merge):
    """Return the built ``IRGraph`` for ``model`` through the production config gate."""
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    return IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024,
        onchip_residual_merge=onchip_residual_merge,
    ).map(repr_)


# --- 1. DEFAULT OFF is byte-identical to the Tier-0 host-add deployment --------


class TestTier1DefaultOffIsByteIdentical:
    def test_default_off_keeps_host_compute_op_add(self):
        ir = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=False)
        assert len(_add_compute_ops(ir)) == 1, (
            "default (flag off) MUST keep the Tier-0 host ComputeOp add"
        )

    def test_default_off_node_structure_matches_no_flag_baseline(self):
        """Flag-off IR node kinds match a baseline IRMapping constructed with no flag
        at all (the byte-identical Tier-0 path)."""
        flow = convert_torch_model(_OnChipSkipResidual(), INPUT_SHAPE, NUM_CLASSES, device="cpu")
        repr_ = flow.get_mapper_repr()
        repr_.assign_perceptron_indices()
        compute_per_source_scales(repr_)
        baseline = IRMapping(
            q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024,
        ).map(repr_)

        ir_off = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=False)
        n_base = len([n for n in baseline.nodes if isinstance(n, NeuralCore)])
        c_base = len([n for n in baseline.nodes if isinstance(n, ComputeOp)])
        n_off = len([n for n in ir_off.nodes if isinstance(n, NeuralCore)])
        c_off = len([n for n in ir_off.nodes if isinstance(n, ComputeOp)])
        assert (n_off, c_off) == (n_base, c_base)

    def test_default_off_deploys_identically_to_no_flag_baseline(self):
        """The flag-off deployed HCM output equals a no-flag baseline deployment of
        the SAME weights bit-for-bit (default path provably unchanged). The baseline
        is built omitting the flag argument entirely (the literal Tier-0 call)."""
        import copy

        torch.manual_seed(7)
        model = _OnChipSkipResidual()
        x = _samples(8, 11).float()
        hcm_off, _ = _build_lif_hcm(copy.deepcopy(model), onchip_residual_merge=False)
        hcm_base, _ = _build_lif_hcm_no_flag(copy.deepcopy(model))
        with torch.no_grad():
            out_off = hcm_off(x)
            out_base = hcm_base(x)
        assert torch.equal(out_off, out_base)


# --- 2. Tier-1 flag-on DEPLOYS as a single on-chip neural segment ---------------


class TestTier1Deploys:
    def test_flag_on_lowers_add_onto_one_extra_neural_core(self):
        ir_off = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=False)
        ir_on = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=True)
        assert len(_add_compute_ops(ir_on)) == 0, "flag on MUST remove the host add"
        n_off = len([n for n in ir_off.nodes if isinstance(n, NeuralCore)])
        n_on = len([n for n in ir_on.nodes if isinstance(n, NeuralCore)])
        assert n_on == n_off + 1, f"flag on adds one merge NeuralCore ({n_off} -> {n_on})"

    def test_onchip_merge_is_single_neural_segment(self):
        """Tier-0 host add splits the model into 2 neural segments (a sync point);
        the Tier-1 on-chip merge keeps the residual on-chip in ONE segment."""
        hcm_off, hyb_off = _build_lif_hcm(_OnChipSkipResidual(), onchip_residual_merge=False)
        hcm_on, hyb_on = _build_lif_hcm(_OnChipSkipResidual(), onchip_residual_merge=True)
        assert mapping_structure(hyb_off)["neural_segments"] == 2, (
            "Tier-0 host add is a cross-segment sync-point merge (2 segments)"
        )
        assert mapping_structure(hyb_on)["neural_segments"] == 1, (
            "Tier-1 on-chip merge must keep the residual on-chip in a single segment"
        )

    def test_onchip_merge_adds_no_host_params(self):
        from mimarsinan.mapping.verification.onchip_majority import count_host_params

        graph_off = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=False)
        graph_on = _ir_for(_OnChipSkipResidual(), onchip_residual_merge=True)
        assert int(count_host_params(graph_on)) == int(count_host_params(graph_off)), (
            "an on-chip merge must not add host params (it removes the already "
            "param-free host add)"
        )


# --- 3. MEASURE the bounded ~1/T deployed delta (Tier-1 vs Tier-0) -------------

# The deployed difference is the SUM of the two characterized components, each
# <= one spike: (A) the unfixed skip-window truncation in the shared HCM input
# fill (~1 spike, closeable outside this unit) + (B) the intrinsic in-segment
# IF-head re-quantization (~1 spike, unreachable by construction). The bound is
# therefore <= 2 spikes per output. A genuine cascade misalignment would drift
# unbounded toward full scale (T spikes) and trip this — the measurement
# distinguishes "bounded re-quant" from "blow-up".
_REQUANT_SPIKE_BOUND = 2.0


class TestTier1DeployedDeltaIsBoundedByOneOverT:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_deployed_delta_vs_tier0_is_within_the_requant_bound(self, seed):
        torch.manual_seed(seed)
        model = _OnChipSkipResidual()
        x = _samples(16, 100 + seed).float()

        hcm_off, _ = _build_lif_hcm(model, onchip_residual_merge=False)
        hcm_on, _ = _build_lif_hcm(model, onchip_residual_merge=True)
        with torch.no_grad():
            out_off = hcm_off(x).double()   # spike counts in [0, T]
            out_on = hcm_on(x).double()

        delta = (out_off - out_on).abs()
        max_delta = float(delta.max())
        assert max_delta <= _REQUANT_SPIKE_BOUND + 1e-9, (
            f"deployed Tier-1-vs-Tier-0 delta {max_delta} spikes exceeds the "
            f"characterized re-quant bound {_REQUANT_SPIKE_BOUND}; this magnitude "
            f"means a real unbounded cascade misalignment, NOT the ~1/T re-quant"
        )
        # And the value-domain (rate) delta is bounded by ~2/T.
        rate_delta = max_delta / float(T)
        assert rate_delta <= 2.0 / float(T) + 1e-9

    def test_tier1_is_not_bit_exact_to_tier0_on_some_sample(self):
        """Honesty lock: Tier-1 is a DIFFERENT valid deployment, not a re-derivation
        of Tier-0 — the intrinsic IF re-quant makes them differ on at least one of a
        spread of seeds. (If a future change made them bit-identical, the merge would
        have silently become the host round-trip again — that must be noticed.)"""
        any_difference = False
        for seed in range(6):
            torch.manual_seed(seed)
            model = _OnChipSkipResidual()
            x = _samples(16, 200 + seed).float()
            hcm_off, _ = _build_lif_hcm(model, onchip_residual_merge=False)
            hcm_on, _ = _build_lif_hcm(model, onchip_residual_merge=True)
            with torch.no_grad():
                d = float((hcm_off(x).double() - hcm_on(x).double()).abs().max())
            if d > 0.0:
                any_difference = True
                break
        assert any_difference, (
            "Tier-1 on-chip merge produced byte-identical output to Tier-0 host-add "
            "across all seeds — the intrinsic in-segment IF re-quant should differ "
            "by ~1 spike somewhere; a perfect match means the merge is no longer "
            "the in-segment on-chip path"
        )

    @pytest.mark.parametrize("t_steps", [4, 8, 16, 32])
    def test_count_delta_is_one_spike_so_value_delta_is_one_over_T(self, t_steps):
        """The MEASURED law: the deployed Tier-1-vs-Tier-0 delta is a fixed ~1 spike
        in COUNT space across T (4..32), so the VALUE (rate = count/T) delta scales
        as ~1/T and SHRINKS with T. A blow-up would scale toward T spikes (full
        scale) and leave the rate delta constant — this test would then trip.
        Worst case over a seed spread is taken (most seeds hit exactly 1 spike;
        one or two hit 0)."""
        worst_count = 0.0
        for seed in range(8):
            torch.manual_seed(seed)
            model = _OnChipSkipResidual()
            x = _samples(16, 100 + seed).float()
            hcm_off, _ = _build_lif_hcm(model, onchip_residual_merge=False, t_steps=t_steps)
            hcm_on, _ = _build_lif_hcm(model, onchip_residual_merge=True, t_steps=t_steps)
            with torch.no_grad():
                d = float((hcm_off(x).double() - hcm_on(x).double()).abs().max())
            worst_count = max(worst_count, d)
        # The count delta is a fixed small number of spikes (<= 2 = A + B), and the
        # value delta tracks worst_count / T — so it shrinks as T grows.
        assert worst_count <= _REQUANT_SPIKE_BOUND + 1e-9, (
            f"T={t_steps}: count delta {worst_count} exceeds the {_REQUANT_SPIKE_BOUND}-spike "
            f"re-quant bound — value would NOT shrink with T (a blow-up signature)"
        )
        value_delta = worst_count / float(t_steps)
        assert value_delta <= _REQUANT_SPIKE_BOUND / float(t_steps) + 1e-9
