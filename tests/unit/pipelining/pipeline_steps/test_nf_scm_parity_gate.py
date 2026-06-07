"""NF↔SCM per-neuron parity gate for analytic TTFS schedules (rung 1 ↔ rung 2)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_nf_scm_parity_or_raise,
    compare_normalized_records,
    nf_scm_parity_enabled,
)

T = 4


class _ToyFlow(nn.Module):
    def __init__(self, mapper_repr, perceptrons):
        super().__init__()
        self._mapper_repr = mapper_repr
        self._perceptrons = nn.ModuleList(perceptrons)

    def get_perceptrons(self):
        return list(self._perceptrons)

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


def _build_aligned_toy(seed=0):
    torch.manual_seed(seed)
    p1 = Perceptron(6, 8, normalization=nn.Identity(), base_activation_name="ReLU")
    p2 = Perceptron(4, 6, normalization=nn.Identity(), base_activation_name="ReLU")
    for p in (p1, p2):
        activation = TTFSCycleActivation(T=T, activation_scale=1.0)
        p.base_activation = activation
        p.activation = activation
        p.set_activation_scale(1.0)

    inp = InputMapper((8,))
    m1 = PerceptronMapper(inp, p1)
    m2 = PerceptronMapper(m1, p2)
    repr_ = ModelRepresentation(m2)
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    ir_graph = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    ).map(repr_)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)
    IRLatency(ir_graph).calculate()

    model = _ToyFlow(repr_, [p1, p2]).double().eval()
    return model, ir_graph


def _pipeline(schedule=None, spiking_mode="ttfs_quantized"):
    p = MockPipeline()
    p.config["spiking_mode"] = spiking_mode
    p.config["firing_mode"] = "TTFS"
    p.config["spike_generation_mode"] = "TTFS"
    p.config["thresholding_mode"] = "<="
    p.config["simulation_steps"] = T
    if schedule is not None:
        p.config["ttfs_cycle_schedule"] = schedule
    return p


class TestEnablement:
    @pytest.mark.parametrize("mode,schedule,enabled", [
        # ttfs_quantized NF deliberately trains the floor-staircase +
        # half-step-bias convention (apply_ttfs_quantization_bias_compensation),
        # which matches the chip ceil kernel only within one step per layer —
        # per-neuron equality is NOT its invariant (46% step-flip fraction on a
        # healthy mmixcore run with a 0.2 pp accuracy gap).
        ("ttfs_quantized", None, False),
        ("ttfs", None, True),
        ("ttfs_cycle_based", "synchronized", True),
        # cascaded: decision-level gate (argmax agreement vs the genuine
        # identity executor) — per-logit atol is meaningless through the host
        # classifier, and WQ tie flips make per-neuron exactness unattainable
        # on real models (driver==executor is bit-exact in clean arithmetic,
        # locked by test_ttfs_segment_node_recorder).
        ("ttfs_cycle_based", "cascaded", True),
        ("lif", None, False),
        ("rate", None, False),
    ])
    def test_gate_runs_only_for_analytic_schedules(self, mode, schedule, enabled):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        cfg = _pipeline(schedule, spiking_mode=mode).config
        contract = SpikingDeploymentContract.from_pipeline_config(cfg)
        assert nf_scm_parity_enabled(contract) is enabled


class TestCascadedDecisionAgreement:
    """The cascaded gate compares decisions (argmax) against the genuine
    identity executor; per-neuron exactness is unattainable on WQ'd real models
    (threshold-normalization tie flips at S-grid boundaries)."""

    class _EchoFlow(torch.nn.Module):
        def __init__(self, model, T, flip=False):
            super().__init__()
            self._model = model
            self._T = T
            self._flip = flip

        def forward(self, x):
            logits = self._model(x) * self._T
            return -logits if self._flip else logits

    def _setup(self, monkeypatch, flip):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        model = torch.nn.Linear(8, 4).double()
        monkeypatch.setattr(
            parity_mod, "_build_cascaded_identity_executor",
            lambda pipeline, model_, ir_graph: self._EchoFlow(
                model_, T, flip=flip,
            ),
        )
        return model

    def test_agreement_passes_when_decisions_match(self, monkeypatch):
        from mimarsinan.pipelining.core.nf_scm_parity import (
            assert_cascaded_nf_scm_agreement_or_raise,
        )

        model = self._setup(monkeypatch, flip=False)
        torch.manual_seed(0)
        samples = torch.rand(16, 8, dtype=torch.float64)
        agreement = assert_cascaded_nf_scm_agreement_or_raise(
            _pipeline("cascaded", spiking_mode="ttfs_cycle_based"),
            model, ir_graph=object(), samples=samples, min_agreement=0.85,
        )
        assert agreement == 1.0

    def test_agreement_fails_loud_on_decision_drift(self, monkeypatch):
        from mimarsinan.pipelining.core.nf_scm_parity import (
            assert_cascaded_nf_scm_agreement_or_raise,
        )

        model = self._setup(monkeypatch, flip=True)
        torch.manual_seed(0)
        samples = torch.rand(16, 8, dtype=torch.float64)
        with pytest.raises(NfScmParityError, match="agreement"):
            assert_cascaded_nf_scm_agreement_or_raise(
                _pipeline("cascaded", spiking_mode="ttfs_cycle_based"),
                model, ir_graph=object(), samples=samples, min_agreement=0.85,
            )


class TestOrderInsensitiveComparison:
    """Conv core emission order ≠ torch flatten order (measured on run
    20260607_045154: p0 bit-exact up to a permutation while its consumers were
    elementwise-exact). The comparison treats each (perceptron, sample) row as
    a multiset; positional wiring is enforced transitively by the consumers'
    exactness and by the rung-3/4 gates."""

    def test_permuted_equal_values_pass(self):
        nf = {0: np.array([[0.0, 0.25, 0.5, 1.0], [0.75, 0.0, 0.25, 0.5]])}
        scm = {0: np.array([[1.0, 0.5, 0.25, 0.0], [0.0, 0.25, 0.5, 0.75]])}
        mismatches, total, _ = compare_normalized_records(nf, scm, atol=1e-9)
        assert (mismatches, total) == (0, 8)

    def test_value_divergence_still_trips(self):
        nf = {0: np.array([[0.0, 0.25, 0.5]])}
        scm = {0: np.array([[0.0, 0.25, 0.75]])}
        mismatches, total, worst = compare_normalized_records(nf, scm, atol=1e-9)
        assert mismatches == 1 and total == 3
        assert worst[0] == pytest.approx(0.25)

    def test_multiset_shift_counts_every_moved_step(self):
        # One neuron moving a step changes two sorted ranks at most; a
        # systematic shift of all neurons trips on every rank.
        nf = {0: np.array([[0.25, 0.25, 0.25, 0.25]])}
        scm = {0: np.array([[0.5, 0.5, 0.5, 0.5]])}
        mismatches, _, _ = compare_normalized_records(nf, scm, atol=1e-9)
        assert mismatches == 4


class TestStepWiring:
    """SoftCoreMappingStep runs the gate for analytic schedules only."""

    class _StubTrainer:
        def __init__(self, batch):
            self._batch = batch

        def iter_validation_batches(self, n):
            yield self._batch, None

    def _make_step(self, schedule=None, spiking_mode="ttfs_quantized", **config):
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        pipeline = _pipeline(schedule, spiking_mode=spiking_mode)
        pipeline.config.update(config)
        step = SoftCoreMappingStep(pipeline)
        step.trainer = self._StubTrainer(torch.rand(8, 8, dtype=torch.float64))
        return step

    def test_analytic_schedule_invokes_gate(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []

        def _record(*args, **kwargs):
            calls.append((args, kwargs))
            return 0.0

        monkeypatch.setattr(parity_mod, "assert_nf_scm_parity_or_raise", _record)
        step = self._make_step(spiking_mode="ttfs")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert len(calls) == 1
        (_, model_arg, _, samples), _kwargs = calls[0]
        assert samples.shape[0] == 2  # nf_scm_parity_samples default

    def test_cascaded_schedule_uses_decision_gate(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        per_neuron_calls = []
        decision_calls = []
        monkeypatch.setattr(
            parity_mod, "assert_nf_scm_parity_or_raise",
            lambda *args, **kwargs: per_neuron_calls.append(1) or 0.0,
        )

        def _decision(*args, **kwargs):
            decision_calls.append(kwargs)
            return 1.0

        monkeypatch.setattr(
            parity_mod, "assert_cascaded_nf_scm_agreement_or_raise", _decision,
        )
        step = self._make_step("cascaded", spiking_mode="ttfs_cycle_based")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert per_neuron_calls == []
        assert len(decision_calls) == 1
        # Healthy cascaded agreement is 1.0 once bias references stay live
        # (the stale-bias incident measured 0.85); default budget is tight.
        assert decision_calls[0]["min_agreement"] == pytest.approx(0.98)

    def test_synchronized_default_budget_is_tight(self, monkeypatch):
        """Synchronized NF is the deployment kernel + segment-entry q(x):
        measured bit-exact (0/122880 on run 20260607_045154), so its default
        budget only needs dtype-tie headroom."""
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []

        def _record(*args, **kwargs):
            calls.append(kwargs)
            return 0.0

        monkeypatch.setattr(parity_mod, "assert_nf_scm_parity_or_raise", _record)
        step = self._make_step("synchronized", spiking_mode="ttfs_cycle_based")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls[0]["max_mismatch_fraction"] == pytest.approx(0.02)

        calls.clear()
        step = self._make_step(spiking_mode="ttfs")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls[0]["max_mismatch_fraction"] == pytest.approx(0.25)

    def test_ttfs_quantized_skips_gate(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_nf_scm_parity_or_raise",
            lambda *args, **kwargs: calls.append(1),
        )
        step = self._make_step(spiking_mode="ttfs_quantized")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls == [], (
            "ttfs_quantized NF trains the floor+half-step-bias convention; "
            "per-neuron equality with the ceil contract is not its invariant"
        )

    def test_zero_samples_disables_gate(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_nf_scm_parity_or_raise",
            lambda *args, **kwargs: calls.append(1),
        )
        step = self._make_step(spiking_mode="ttfs", nf_scm_parity_samples=0)
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls == []


class TestParityGate:
    def test_passes_on_aligned_toy_model_with_zero_fraction(self):
        model, ir_graph = _build_aligned_toy()
        torch.manual_seed(1)
        samples = torch.rand(3, 8, dtype=torch.float64)
        fraction = assert_nf_scm_parity_or_raise(
            _pipeline(), model, ir_graph, samples,
        )
        assert fraction == 0.0

    def test_fails_loud_on_injected_core_perturbation(self):
        model, ir_graph = _build_aligned_toy()
        # Perturb one IR core's weights only — the mapping diverges from NF.
        # (Strong enough to push the victim neuron across the firing threshold.)
        victim = ir_graph.get_neural_cores()[0]
        victim.core_matrix = victim.core_matrix.copy()
        victim.core_matrix[:, 0] += 3.0
        torch.manual_seed(1)
        samples = torch.rand(3, 8, dtype=torch.float64)
        with pytest.raises(NfScmParityError) as excinfo:
            assert_nf_scm_parity_or_raise(
                _pipeline(), model, ir_graph, samples,
            )
        message = str(excinfo.value)
        assert "perceptron" in message and "neuron" in message, (
            f"diff message must name the diverging neuron: {message}"
        )

    def test_stale_instance_forward_fails_with_actionable_message(self):
        """A synchronized model carrying an instance ``forward`` override (the
        pre-schedule-aware-tuner cascade forward) is the incident's signature;
        name the cause instead of reporting raw mismatches."""
        model, ir_graph = _build_aligned_toy()
        model.forward = lambda x: model.get_mapper_repr()(x)
        torch.manual_seed(1)
        samples = torch.rand(3, 8, dtype=torch.float64)
        with pytest.raises(NfScmParityError, match="instance forward"):
            assert_nf_scm_parity_or_raise(
                _pipeline("synchronized", spiking_mode="ttfs_cycle_based"),
                model, ir_graph, samples,
            )

    def test_mismatch_budget_tolerates_isolated_flips(self):
        model, ir_graph = _build_aligned_toy()
        victim = ir_graph.get_neural_cores()[0]
        victim.core_matrix = victim.core_matrix.copy()
        victim.core_matrix[:, 0] += 3.0
        torch.manual_seed(1)
        samples = torch.rand(3, 8, dtype=torch.float64)
        # A full budget swallows everything — documents the knob's semantics.
        assert_nf_scm_parity_or_raise(
            _pipeline(), model, ir_graph, samples,
            max_mismatch_fraction=1.0,
        )
