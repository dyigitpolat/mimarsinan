"""Streamed exactness ACROSS a mixed wire/absolute seam: NF↔SCM window counts
hold at atol=0 when a host residual re-joins the branch that crossed a core.

This is the deployed half of the gauge-establishment contract. The twin decodes
each source of the join at its producer's gauge through the
``ScaleNormalizingWrapper`` its wrap slots describe; IR emission installs the
SAME wrapper and the hybrid executor runs it. If those two ever drifted apart,
the residual would arrive in the wrong currency and the counts would part — so
the atol=0 gate is what turns "one definition" into a measured fact.

Two things have to hold for that to be a measurement rather than a ritual, and
both are asserted below rather than assumed:

1. The seam value must REACH the comparison. The gate compares per-PERCEPTRON
   window counts, so a fixture whose seam has no neural consumer cannot observe
   the seam at all. ``_ResidualStem`` therefore puts a CONSUMING core downstream
   of the residual, exactly as the real transformer does (the next block reads
   it), and ``test_the_seam_feeds_a_neural_core`` pins that structurally.
2. The cascade must be LIVE. A nearly silent cascade absorbs any currency error
   below its firing threshold: measured, a default-init d=8 stem kept every
   count bit-identical under both SSOT breaks. The gains below are chosen so
   both cores' counts span the window, and ``test_both_cores_are_live`` pins it.

The three break tests are the teeth: each forces ONE side of the "one
definition" claim to disagree — the twin drops the wrapper, IR emission drops
it, or the deployed wrapper decodes every source at unity instead of at its
producer's gauge — and the gate must fire on each.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.pruning import derive_deployed_neuron_survival
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.pipelining.core import nf_scm_parity
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_streamed_nf_scm_exact_or_raise,
)
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW

T = 8
D = 16
INPUT_SHAPE = (1, 1, D)
NUM_CLASSES = 4
# Distinct thetas: the seam's two sources must carry DIFFERENT currencies, or
# the wrapper is numerically inert and no break is observable.
THETAS = (3.0, 1.5)


class _ResidualStem(nn.Module):
    """The transformer block shape at minimum size: the stem's value skips the
    neural core and re-joins after a host Linear — and the NEXT block's core
    consumes the join, which is what carries the seam into a window count."""

    def __init__(self, d: int = D, n: int = NUM_CLASSES):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, d)
        self.act2 = nn.ReLU()
        self.head = nn.Linear(d, n)

    def forward(self, x):
        x = x.flatten(1)
        x = x + self.fc2(self.act(self.fc1(self.norm(x))))
        return self.head(self.act2(self.fc3(x)))


class _StreamedNFModel(nn.Module):
    """The pipeline's post-adaptation shape: ``forward`` IS the per-segment
    raw-cascade walk ``_ChipAlignedNFForward`` installs."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.repr_ = flow.get_mapper_repr()
        self._perceptrons = nn.ModuleList(list(self.repr_.get_perceptrons()))

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(self.repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))(x)


def _streamed_pipeline_stub():
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "spiking_family": "lif",
        "spiking_variant": "streamed",
        "simulation_steps": T,
        "input_shape": INPUT_SHAPE,
        "device": "cpu",
    })
    return SimpleNamespace(config=cfg)


def _seam_model():
    torch.manual_seed(0)
    stem = _ResidualStem()
    with torch.no_grad():
        # Drive both cores into their firing range (see the module docstring):
        # a silent cascade cannot observe a currency error at all.
        stem.fc1.weight.mul_(2.0)
        stem.fc1.bias.fill_(0.5)
        stem.fc2.weight.mul_(2.0)
        stem.fc2.bias.fill_(0.2)
        stem.fc3.weight.mul_(3.0)
        stem.fc3.bias.fill_(0.3)
    flow = convert_torch_model(
        stem, INPUT_SHAPE, NUM_CLASSES, device="cpu",
        encoding_layer_placement="offload",
    ).eval()
    repr_ = flow.get_mapper_repr()
    assert heterogeneous_domain_joins(repr_), "fixture must carry the mixed seam"
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(THETAS[i]))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif

    establish_gauge_for_mixed_domain_seams(flow, input_data_scale=1.0)
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64,
    ).map(repr_)
    return flow, repr_, ir


def _samples(n=4):
    torch.manual_seed(1)
    return torch.rand(n, *INPUT_SHAPE)


def _seam_node(repr_):
    return next(
        n for n in repr_.execution_order() if getattr(n, "name", None) == "add"
    )


def _window_counts(flow, ir):
    """``(nf, scm)`` per-perceptron window counts — the two sides the gate
    compares, exposed so a test can assert the fixture is actually live."""
    model = _StreamedNFModel(flow)
    samples = _samples()
    executor = nf_scm_parity._build_streamed_identity_executor(
        _streamed_pipeline_stub(), model, ir,
    )
    per_sample = []
    with measurement_plane():
        nf = nf_scm_parity._capture_nf_streamed_counts(model, samples)
        nf = derive_deployed_neuron_survival(ir).project(nf)
        with torch.no_grad():
            for i in range(samples.shape[0]):
                _, record = executor.forward_with_recording(
                    samples[i : i + 1], sample_index=i,
                )
                per_sample.append(nf_scm_parity._group_record_by_perceptron(
                    record, executor.hybrid_mapping,
                    values_of=lambda core: core.output_spike_count[: core.n_out_used],
                ))
    scm = {pi: np.stack([s[pi] for s in per_sample]) for pi in per_sample[0]}
    return {pi: np.rint(nf[pi]) for pi in scm}, scm


def _seam_wrappers(ir):
    for op in ir.nodes:
        module = (op.params or {}).get("module") if isinstance(op, ComputeOp) else None
        if isinstance(module, ScaleNormalizingWrapper):
            yield op, module


class TestStreamedExactnessAtAMixedSeam:
    def test_the_seam_feeds_a_neural_core(self):
        """The gate compares PER-PERCEPTRON window counts, so a seam with no
        neural consumer is invisible to it. Pin the consumer structurally: this
        is the assertion that keeps the fixture able to fail at all."""
        _, repr_, _ = _seam_model()
        consumers = repr_.consumer_map()
        frontier = list(consumers.get(id(_seam_node(repr_)), []))
        seen, reached = set(), []
        while frontier:
            node = frontier.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            perceptron = getattr(node, "perceptron", None)
            if perceptron is not None:
                reached.append(perceptron)
            else:
                frontier.extend(consumers.get(id(node), []))
        assert reached, (
            "the mixed seam must have a downstream perceptron, or its value "
            "never reaches a per-perceptron window count and the exactness "
            "gate below is vacuous"
        )

    def test_both_cores_are_live(self):
        """Live means the counts vary: a core pinned at 0 (or at T) absorbs any
        currency error below its firing threshold, and the break tests would
        pass for the wrong reason."""
        flow, _, ir = _seam_model()
        nf, _ = _window_counts(flow, ir)
        assert sorted(nf) == [0, 1], "both cores must reach the comparison"
        for pi, counts in nf.items():
            assert len(np.unique(counts)) >= 4, (
                f"perceptron {pi} counts are degenerate ({np.unique(counts)}): "
                "a silent cascade cannot detect a currency break"
            )

    def test_deployed_seam_op_is_the_twin_s_own_wrapper(self):
        """One definition, not two: the module IR emission installs at the seam
        IS the composition the twin's host-value forward runs."""
        _, repr_, ir = _seam_model()
        seam = _seam_node(repr_)
        assert isinstance(seam._maybe_wrap_for_scales(), ScaleNormalizingWrapper)
        multi = [m for _, m in _seam_wrappers(ir) if m._num_inputs > 1]
        assert multi, "the deployed join must carry the per-source wrapper"

    def test_window_counts_exact_across_the_mixed_seam(self):
        flow, _, ir = _seam_model()
        assert_streamed_nf_scm_exact_or_raise(
            _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
        )

    def test_gate_has_teeth_on_this_fixture(self):
        """A threshold drift must trip the atol=0 gate here, so the exactness
        above is a measurement and not a vacuous pass."""
        flow, _, ir = _seam_model()
        core = ir.get_neural_cores()[-1]
        core.threshold = float(core.threshold) * 2.0
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
            )

    def test_gate_catches_a_twin_that_skips_the_emitted_wrapper(self, monkeypatch):
        """Break #1, TWIN side: the driver runs the bare module instead of the
        emitted ``ScaleNormalizingWrapper``, so the seam's sources never decode
        at their producers' gauges. The deployed side is untouched."""
        flow, _, ir = _seam_model()
        monkeypatch.setattr(
            SegmentForwardDriver, "_host_value_forward", lambda self, node: None,
        )
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
            )

    def test_gate_catches_emission_that_drops_the_wrapper(self):
        """Break #2, DEPLOYED side: IR emission installs the bare module, so
        the chip computes the join on mixed currencies while the twin still
        transcodes. The twin is untouched."""
        flow, _, ir = _seam_model()
        stripped = 0
        for op, module in list(_seam_wrappers(ir)):
            op.params["module"] = module.module
            stripped += 1
        assert stripped, "fixture must emit wrappers to strip"
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
            )

    def test_gate_catches_a_seam_decoded_at_unity(self):
        """Break #3, the GAUGE CHOICE itself: the deployed join decodes every
        source at unity instead of at its producer's gauge. This is the §11.2
        physics claim — kappa_T == kappa_S — measured rather than asserted in a
        docstring."""
        flow, _, ir = _seam_model()
        retuned = 0
        for _, module in _seam_wrappers(ir):
            if module._num_inputs <= 1:
                continue
            for i in range(module._num_inputs):
                buffer = getattr(module, f"input_scale_{i}")
                setattr(module, f"input_scale_{i}", torch.ones_like(buffer))
            retuned += 1
        assert retuned, "fixture must carry a multi-source seam wrapper"
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
            )

    def test_owns_scale_domain_is_not_the_streamed_lever(self):
        """Honesty pin. ``compute_op_owns_scale_domain`` looks like the deployed
        seam's SSOT, but the rate/LIF host path passes ``apply_ttfs=False`` and
        so takes the ``(1, 1)`` early return before ever consulting it — which
        is why forcing it False leaves streamed counts untouched. State that
        here so nobody mistakes a passing gate for coverage of this predicate;
        the levers that DO carry the streamed seam are the three breaks above.
        """
        from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
            resolve_stage_compute_scales,
        )

        _, _, ir = _seam_model()
        op, _ = next(iter(_seam_wrappers(ir)))
        assert resolve_stage_compute_scales(
            ir, op.id, apply_ttfs=False, op=op,
        ) == (1.0, 1.0)
