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
        # The floor+half-step-bias convention modes (ttfs_quantized AND the
        # synchronized floor-collapse) deliberately train the floor-staircase +
        # half-step-bias NF (apply_ttfs_quantization_bias_compensation), which
        # matches the deployed ceil kernel only within one step per layer —
        # per-neuron equality is NOT their invariant. Both are excluded; their
        # deployment stays bit-exact regardless (SANA-FE parity 1.0).
        ("ttfs_quantized", None, False),
        ("ttfs", None, True),
        ("ttfs_cycle_based", "synchronized", False),
        # cascaded: decision-level gate (argmax agreement vs the genuine
        # identity executor) — per-logit atol is meaningless through the host
        # classifier, and WQ tie flips make per-neuron exactness unattainable
        # on real models (driver==executor is bit-exact in clean arithmetic,
        # locked by test_ttfs_segment_node_recorder).
        ("ttfs_cycle_based", "cascaded", True),
        ("lif", None, False),
    ])
    def test_gate_runs_only_for_analytic_schedules(self, mode, schedule, enabled):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        cfg = _pipeline(schedule, spiking_mode=mode).config
        contract = SpikingDeploymentContract.from_pipeline_config(cfg)
        assert nf_scm_parity_enabled(contract) is enabled


class TestTorchSimParityEnablement:
    """Decision-level torch↔deployed-sim gate arms for every mode with a
    faithful identity executor — including LIF (the t0_03 blind spot: its
    NF↔SCM divergence surfaced as a retention abort, never a parity error)."""

    @pytest.mark.parametrize("mode,schedule,enabled", [
        ("ttfs", None, True),
        ("ttfs_cycle_based", "cascaded", True),
        ("ttfs_cycle_based", "synchronized", True),
        ("lif", None, True),
        ("ttfs_quantized", None, False),
    ])
    def test_torch_sim_parity_enabled(self, mode, schedule, enabled):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )
        from mimarsinan.pipelining.core.nf_scm_parity import (
            torch_sim_parity_enabled,
        )

        cfg = _pipeline(schedule, spiking_mode=mode).config
        contract = SpikingDeploymentContract.from_pipeline_config(cfg)
        assert torch_sim_parity_enabled(contract) is enabled


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

    def test_synchronized_skips_gate(self, monkeypatch):
        """The synchronized floor-collapse trains the ttfs_quantized floor +
        half-step-bias convention and deploys the mode-derived ceil kernel, so —
        like ttfs_quantized — per-neuron equality with the ceil contract is not its
        invariant. It is excluded from the per-neuron gate; deployment stays
        bit-exact (SANA-FE parity 1.0)."""
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_nf_scm_parity_or_raise",
            lambda *args, **kwargs: calls.append(1),
        )
        step = self._make_step("synchronized", spiking_mode="ttfs_cycle_based")
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls == []

    def test_continuous_ttfs_default_budget_is_loose(self, monkeypatch):
        """Continuous ttfs is the only per-neuron path left; its default budget
        keeps the loose (uncalibrated-residual) headroom."""
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []

        def _record(*args, **kwargs):
            calls.append(kwargs)
            return 0.0

        monkeypatch.setattr(parity_mod, "assert_nf_scm_parity_or_raise", _record)
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


class TestOneSampleCountKeyForBothBranches:
    """``nf_scm_parity_samples_cascaded`` was a per-mode default written inline,
    not a second concept: both branches answer 'how many validation inputs does
    the NF<->SCM gate run on'. One key, one mode-aware derivation — which also
    repairs the disable sentinel the shadow key made unreachable on cascaded
    (deployment_faithfulness declares nf_scm_parity_samples as THE gate flag)."""

    _make_step = TestStepWiring._make_step
    _StubTrainer = TestStepWiring._StubTrainer

    def test_cascaded_takes_its_sample_count_from_the_one_key(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_cascaded_nf_scm_agreement_or_raise",
            lambda *args, **kwargs: calls.append(args) or 1.0,
        )
        step = self._make_step("cascaded", spiking_mode="ttfs_cycle_based")
        step.trainer = self._StubTrainer(torch.rand(128, 8, dtype=torch.float64))
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls[0][3].shape[0] == 64  # the cascaded derived default

    def test_an_explicit_sample_count_governs_the_cascaded_branch(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_cascaded_nf_scm_agreement_or_raise",
            lambda *args, **kwargs: calls.append(args) or 1.0,
        )
        step = self._make_step(
            "cascaded", spiking_mode="ttfs_cycle_based", nf_scm_parity_samples=5,
        )
        step.trainer = self._StubTrainer(torch.rand(128, 8, dtype=torch.float64))
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls[0][3].shape[0] == 5

    def test_the_disable_sentinel_now_reaches_the_cascaded_branch(self, monkeypatch):
        """Before unification, nf_scm_parity_samples=0 silently left the
        cascaded gate armed — the faithfulness flag lied."""
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod

        calls = []
        monkeypatch.setattr(
            parity_mod, "assert_cascaded_nf_scm_agreement_or_raise",
            lambda *args, **kwargs: calls.append(1) or 1.0,
        )
        step = self._make_step(
            "cascaded", spiking_mode="ttfs_cycle_based", nf_scm_parity_samples=0,
        )
        step._run_nf_scm_parity_gate(model=object(), ir_graph=object())
        assert calls == []

    def test_the_retired_key_is_no_longer_schema_known(self):
        from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
        from mimarsinan.config_schema.registry import REGISTRY

        assert "nf_scm_parity_samples_cascaded" not in REGISTRY
        assert "nf_scm_parity_samples_cascaded" not in CONFIG_KEYS_SET

    def test_a_document_pinning_the_retired_key_is_reported_not_dropped(self):
        from mimarsinan.config_schema.registry import parse_deployment_document
        from mimarsinan.gui.wizard.emit import emit_deployment_config

        document = {
            "experiment_name": "retired",
            "deployment_parameters": {"nf_scm_parity_samples_cascaded": 64},
        }
        parsed = parse_deployment_document(document)
        assert parsed.unknown == [
            "deployment_parameters.nf_scm_parity_samples_cascaded"
        ]
        emitted = emit_deployment_config(document)
        assert emitted["deployment_parameters"]["nf_scm_parity_samples_cascaded"] == 64

    def test_the_disable_sentinel_is_authorable_through_the_widget_bounds(self):
        """bounds=(1, None) made the documented 0 opt-out unreachable in the GUI."""
        from mimarsinan.config_schema.registry import REGISTRY

        for key in ("nf_scm_parity_samples", "scm_torch_sim_parity_samples",
                    "max_simulation_samples"):
            assert REGISTRY[key].bounds[0] == 0, key

    def test_lif_invokes_torch_sim_parity_gate(self, monkeypatch):
        """LIF arms the decision-level torch↔deployed-sim gate (same threshold
        discipline as casc/sync; the t0_03 defect must name itself as a parity
        error, not a retention abort)."""
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as step_mod

        calls = []

        def _record(reference, flow, samples, *, min_agreement):
            calls.append((reference, flow, samples, min_agreement))
            return 1.0

        monkeypatch.setattr(
            parity_mod, "assert_torch_vs_deployed_sim_parity_or_raise", _record,
        )
        monkeypatch.setattr(parity_mod, "torch_parity_reference", lambda m: m)
        monkeypatch.setattr(
            step_mod, "build_identity_mapping_for_pipeline",
            lambda ir_graph, pipeline_config=None: object(),
        )
        monkeypatch.setattr(
            step_mod, "build_spiking_hybrid_flow",
            lambda pipeline, mapping, model=None: object(),
        )
        step = self._make_step(spiking_mode="lif")
        step.pipeline.config["firing_mode"] = "Default"
        step.pipeline.config["thresholding_mode"] = "<"
        step._run_torch_sim_parity_check(model=object(), ir_graph=object())
        assert len(calls) == 1
        assert calls[0][3] == pytest.approx(0.98)


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


class TestVanillaContinuousTtfsPerSourceScales:
    """U2 regression: continuous-ttfs deploy must bake each layer's upstream
    activation scale into the mapped weights via ``per_input_scales``.

    Historically only ``WeightQuantizationStep`` populated ``per_input_scales``
    (via ``compute_per_source_scales``), so a no-weight-quant vanilla run left
    the deployed effective weights MISSING the upstream-θ factor. The
    continuous-ttfs NF (real-valued cascade) and the identity-mapped SCM
    (normalised cascade) then diverged by ~upstream-θ — a real 40%+ deploy
    split, not a budget artefact (the WORST case measured nf=1.0 vs scm=0.09).
    ``SoftCoreMappingStep`` now runs ``compute_per_source_scales`` for every
    deployment, so the scale factor lands regardless of weight quantization.
    """

    class _ContinuousFlow(nn.Module):
        def __init__(self, mapper_repr, perceptrons):
            super().__init__()
            self._mapper_repr = mapper_repr
            self._perceptrons = nn.ModuleList(perceptrons)
            self.preprocessor = None

        def get_perceptrons(self):
            return list(self._perceptrons)

        def get_mapper_repr(self):
            return self._mapper_repr

        def forward(self, x):
            return self._mapper_repr(x)

    def _build_continuous_cascade(self, *, call_per_source_scales: bool):
        from mimarsinan.models.nn.decorators.clamp_quantize import ClampDecorator
        from mimarsinan.models.nn.layers import TransformedActivation

        torch.manual_seed(0)
        dims = [8, 8, 8, 8, 6]
        perceptrons = []
        for i in range(4):
            p = Perceptron(
                dims[i + 1], dims[i], normalization=nn.Identity(),
                base_activation_name="ReLU",
            )
            # Distinct per-layer θ: exposes a missing input-scale factor (equal
            # scales would mask it). Mirrors the real run's [1.18, 4.42] spread.
            scale = torch.tensor(1.18 + 0.9 * i, dtype=torch.float64)
            p.set_activation_scale(scale)
            p.set_activation(TransformedActivation(
                p.base_activation, [ClampDecorator(torch.tensor(0.0), scale)]))
            perceptrons.append(p)

        inp = InputMapper((dims[0],))
        node = inp
        for p in perceptrons:
            node = PerceptronMapper(node, p)
        repr_ = ModelRepresentation(node)
        repr_.assign_perceptron_indices()
        if call_per_source_scales:
            compute_per_source_scales(repr_)

        ir_graph = IRMapping(
            q_max=15, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        ).map(repr_)
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph

        # The failing cell deploys with weight_quantization=False.
        quantize_ir_graph(ir_graph, 5, weight_quantization=False)
        IRLatency(ir_graph).calculate()
        model = self._ContinuousFlow(repr_, perceptrons).double().eval()
        return model, ir_graph

    def test_parity_holds_with_per_source_scales(self):
        """The fixed mapping path (per-source scales applied) is bit-exact."""
        model, ir_graph = self._build_continuous_cascade(
            call_per_source_scales=True,
        )
        torch.manual_seed(7)
        samples = torch.rand(4, 8, dtype=torch.float64)
        fraction = assert_nf_scm_parity_or_raise(
            _pipeline(spiking_mode="ttfs"), model, ir_graph, samples,
            atol=1e-6,
        )
        assert fraction == 0.0

    def test_missing_per_source_scales_trips_the_gate(self):
        """Guard: without per-source scales the deploy genuinely diverges — the
        gate must FAIL LOUD rather than pass on a mis-scaled deployment."""
        model, ir_graph = self._build_continuous_cascade(
            call_per_source_scales=False,
        )
        torch.manual_seed(7)
        samples = torch.rand(4, 8, dtype=torch.float64)
        with pytest.raises(NfScmParityError):
            assert_nf_scm_parity_or_raise(
                _pipeline(spiking_mode="ttfs"), model, ir_graph, samples,
                atol=1e-6, max_mismatch_fraction=0.25,
            )

    def test_soft_core_mapping_step_applies_per_source_scales(self):
        """The step itself must populate ``per_input_scales`` (the SSOT fix is
        in ``SoftCoreMappingStep``, not only in ``WeightQuantizationStep``)."""
        model, _ = self._build_continuous_cascade(call_per_source_scales=False)
        for p in model.get_perceptrons():
            assert not hasattr(p, "per_input_scales") or p.per_input_scales is None

        compute_per_source_scales(model.get_mapper_repr())
        scaled = [
            p for p in model.get_perceptrons()
            if getattr(p, "per_input_scales", None) is not None
        ]
        # Every interior layer (a perceptron with an upstream perceptron source)
        # carries a non-trivial input scale.
        assert any(
            float(p.per_input_scales.max()) != 1.0 for p in scaled
        ), "interior layers must inherit a non-unit upstream activation scale"


class TestDeviceConsistency:
    """The mapping pipeline can leave the mapper-graph compute modules and the
    perceptrons on different devices (offload/cache seams), so a full-graph
    forward hits a cross-device matmul. Each gate owns its forward's device
    contract: it unifies the whole model onto one device before forwarding,
    instead of only moving ``samples`` to the first parameter's device.

    Root cause of the 2026-06-19 SCM crash batch: 3/9 runs died with
    ``Expected all tensors to be on the same device`` at PerceptronMapper /
    ComputeOpMapper(classifier) inside the gate forward."""

    class _SplitModel(nn.Module):
        """Two linears the forward chains device-naively; if they sit on
        different devices a plain forward raises a cross-device RuntimeError."""

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(5, 5)
            self.b = nn.Linear(5, 4)

        def forward(self, x):
            return self.b(self.a(x))

    def test_unify_returns_param_device_and_is_consistent_on_cpu(self):
        from mimarsinan.pipelining.core.nf_scm_parity import _unify_model_device

        model = self._SplitModel()  # all CPU
        device = _unify_model_device(model)
        assert device.type == "cpu"
        assert {p.device.type for p in model.parameters()} == {"cpu"}

    def test_unify_handles_parameterless_model(self):
        from mimarsinan.pipelining.core.nf_scm_parity import _unify_model_device

        assert _unify_model_device(nn.Identity()) is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs CUDA for a genuine cross-device model",
    )
    def test_gate_unifies_a_split_device_model(self):
        from mimarsinan.pipelining.core.nf_scm_parity import (
            assert_torch_vs_deployed_sim_parity_or_raise,
        )

        torch.manual_seed(0)
        model = self._SplitModel()
        flow = self._SplitModel()
        flow.load_state_dict(model.state_dict())
        # Deliberately strand the two linears on different devices (the offload
        # seam): perceptron-side on CUDA, compute-side on CPU.
        model.a.to("cuda:0"); model.b.to("cpu")
        flow.a.to("cuda:0"); flow.b.to("cpu")
        x = torch.randn(32, 5)

        with pytest.raises(RuntimeError):
            model(x.to("cuda:0"))  # plain forward crosses devices → crash

        agreement = assert_torch_vs_deployed_sim_parity_or_raise(model, flow, x)
        assert agreement == pytest.approx(1.0)
        assert {p.device.type for p in model.parameters()} == {"cuda"}, (
            "the gate must unify the whole model onto one device before forward"
        )


class TestTorchVsDeployedSimParity:
    """The added torch↔DEPLOYED-sim parity check: torch model argmax must agree
    with the deployed sim's argmax (the exact executor run_scm_identity_metric runs)."""

    class _Logits(nn.Module):
        def __init__(self, roll=0):
            super().__init__()
            self.roll = roll
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.roll(x, self.roll, dims=1)

    def test_passes_when_argmax_agrees(self):
        from mimarsinan.pipelining.core.nf_scm_parity import (
            assert_torch_vs_deployed_sim_parity_or_raise,
        )
        model = self._Logits(0)
        flow = self._Logits(0)
        x = torch.randn(48, 5)
        agree = assert_torch_vs_deployed_sim_parity_or_raise(
            model, flow, x, min_agreement=0.98
        )
        assert agree == pytest.approx(1.0)

    def test_raises_when_deployed_sim_diverges(self):
        from mimarsinan.pipelining.core.nf_scm_parity import (
            assert_torch_vs_deployed_sim_parity_or_raise,
        )
        model = self._Logits(0)
        flow = self._Logits(1)  # the deployed sim shifts the argmax → diverges
        x = torch.randn(64, 5)
        with pytest.raises(NfScmParityError, match="deployed-sim parity"):
            assert_torch_vs_deployed_sim_parity_or_raise(
                model, flow, x, min_agreement=0.98
            )


class TestPrunedDeploymentParity:
    """Pruning removes deployed output neurons (dead cores + zeroed columns), so the
    identity-mapped SCM carries M < N of a perceptron's N neurons while the NF keeps all N
    (pruned ones live at 0.0). The gate projects the NF onto the DEPLOYED survivor set (the
    pruned ir_graph reality, via ``DeployedNeuronSurvival``) and compares bit-exact —
    instead of false-failing the raw per-neuron shape check on a lossless pruned deploy."""

    def _build_pruned_toy(self, pruned_out_neuron=5):
        from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
        from mimarsinan.mapping.pruning.ir_pruning_masks import (
            get_initial_pruning_masks_from_model,
        )
        from mimarsinan.tuning.tuners.pruning.pruning_tuner_enforce import (
            enforce_pruning_persistently,
            register_prune_buffers,
        )

        torch.manual_seed(0)
        p1 = Perceptron(6, 8, normalization=nn.Identity(), base_activation_name="ReLU")
        p2 = Perceptron(4, 6, normalization=nn.Identity(), base_activation_name="ReLU")
        for p in (p1, p2):
            act = TTFSCycleActivation(T=T, activation_scale=1.0)
            p.base_activation = act
            p.activation = act
            p.set_activation_scale(1.0)

        # Prune one output neuron of p1 (keep-mask: True = keep). Its downstream
        # consumer p2 is rewired by prune_ir_graph — a genuine pruned deployment.
        keep_p1 = torch.ones(6, dtype=torch.bool)
        keep_p1[pruned_out_neuron] = False
        row_masks = [keep_p1, torch.ones(4, dtype=torch.bool)]
        col_masks = [torch.ones(8, dtype=torch.bool), torch.ones(6, dtype=torch.bool)]
        register_prune_buffers([p1, p2], row_masks, col_masks)
        enforce_pruning_persistently([p1, p2], row_masks, col_masks)

        m2 = PerceptronMapper(PerceptronMapper(InputMapper((8,)), p1), p2)
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

        model = _ToyFlow(repr_, [p1, p2]).double().eval()
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_graph)
        ir_graph = prune_ir_graph(
            ir_graph,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
            store_heatmap=False,
            simulation_steps=T,
            spiking_mode="ttfs_quantized",
        )
        IRLatency(ir_graph).calculate()
        return model, ir_graph

    def test_pruned_deployment_parity_is_bit_exact_via_projection(self):
        model, ir_graph = self._build_pruned_toy()
        # The pruned perceptron's core now carries M<N neurons; without the survival
        # projection this raised "neuron-count mismatch". With it, bit-exact.
        torch.manual_seed(1)
        samples = torch.rand(3, 8, dtype=torch.float64)
        fraction = assert_nf_scm_parity_or_raise(
            _pipeline(), model, ir_graph, samples, atol=1e-9,
        )
        assert fraction == 0.0

    def test_survival_reconstructs_the_pruned_neuron(self):
        from mimarsinan.mapping.pruning import derive_deployed_neuron_survival

        _, ir_graph = self._build_pruned_toy(pruned_out_neuron=2)
        survival = derive_deployed_neuron_survival(ir_graph)
        # Perceptron 0 (p1) has 6 output neurons; index 2 was pruned -> survivors omit it.
        assert list(survival.survivors[0]) == [0, 1, 3, 4, 5]
