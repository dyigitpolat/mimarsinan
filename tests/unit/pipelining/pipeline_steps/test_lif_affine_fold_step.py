"""LIFAffineFoldStep: scheduling, premise gating, Novena S-gate, fold commit.

R2 (lossless_refinement_ledger.md §2D + §3 R2): the premise teacher is the
post-AQ, PRE-adaptation reference read (``aq_reference_read``), a premise-skip
is a TRUE no-op (the old pre-applied half-step moved skipped runs -0.5..-1.7pp),
and Novena cells are admitted regardless of the AQ flag (the t0_02 class).
"""

from __future__ import annotations

import copy

import torch

from conftest import (
    MockDataProviderFactory,
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.mapping.support.bias_compensation import LIF_HALF_STEP_FLAG
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.adaptation.lif_affine_fold_step import (
    LIFAffineFoldStep,
)
from mimarsinan.tuning.lif_affine_fold import AFFINE_FOLD_FLAG, CraterPremiseVerdict


def _config(**overrides):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 8
    cfg["cycle_accurate_lif_forward"] = True
    cfg["activation_quantization"] = True
    cfg["lif_affine_fold"] = True
    cfg["lif_half_step_bias"] = False
    cfg.update(overrides)
    return cfg


def _deployed_model(cfg, hidden_layers=2):
    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    torch.manual_seed(0)
    model = copy.deepcopy(make_tiny_supermodel(hidden_layers=hidden_layers))
    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5
    tuner = LIFAdaptationTuner(
        pipeline, model, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=AdaptationManager(),
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model


class _RecordingReporter:
    def __init__(self):
        self.reports = {}

    def report(self, key, value, *args, **kwargs):
        self.reports[key] = value

    def console_log(self, *args, **kwargs):
        pass


def _run_step(cfg, model, aq_reference=1.0):
    torch.manual_seed(7)
    factory = MockDataProviderFactory()
    factory.create()  # materialize the dataset under the fixed seed
    pipeline = MockPipeline(config=cfg, data_provider_factory=factory)
    pipeline.reporter = _RecordingReporter()
    pipeline.seed("model", model, step_name="LIF Adaptation")
    pipeline.seed(
        "aq_reference_read", aq_reference, step_name="Activation Quantization",
    )
    step = LIFAffineFoldStep(pipeline)
    step.name = "LIF Affine Fold"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


def _model_snapshot(model):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    scales = [
        None if getattr(p, "per_input_scales", None) is None
        else torch.as_tensor(p.per_input_scales).clone()
        for p in model.get_perceptrons()
    ]
    return state, scales


def _assert_model_matches_snapshot(model, snapshot):
    state_before, scales_before = snapshot
    state_after = model.state_dict()
    assert state_before.keys() == state_after.keys()
    for key in state_before:
        assert torch.equal(state_before[key], state_after[key]), key
    for p, saved in zip(model.get_perceptrons(), scales_before):
        now = getattr(p, "per_input_scales", None)
        if saved is None:
            assert now is None, p.name
        else:
            assert now is not None and torch.equal(torch.as_tensor(now), saved), p.name


class TestAppliesTo:
    def test_off_by_default(self):
        cfg = _config()
        cfg.pop("lif_affine_fold")
        assert not LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))

    def test_on_when_knob_set_for_lif(self):
        assert LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(_config()))

    def test_never_for_ttfs_modes(self):
        cfg = _config(spiking_mode="ttfs_quantized", firing_mode="TTFS",
                      thresholding_mode="<=")
        assert not LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))

    def test_fp_default_stays_out(self):
        # Without AQ the Default-reset chain has no scheduled repair here.
        cfg = _config(activation_quantization=False)
        assert not LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))

    def test_fp_novena_is_admitted(self):
        # [R2c] the Novena C4 repair (the only sanctioned V7 fix) must reach
        # fp cells: t0_02 never scheduled the step (ledger §2D).
        cfg = _config(activation_quantization=False, firing_mode="Novena")
        assert LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))

    def test_wq_only_novena_is_admitted(self):
        cfg = _config(
            activation_quantization=False, weight_quantization=True,
            firing_mode="Novena",
        )
        assert LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))


class TestContract:
    def test_step_requires_the_aq_reference(self):
        # [R2b] the premise teacher entry is a hard class-level requirement:
        # the assembly DAG check then enforces AQ-before-fold on every cell.
        assert "aq_reference_read" in LIFAffineFoldStep.REQUIRES


class TestStepOrdering:
    def test_selected_before_weight_quantization(self):
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
        )

        cfg = _config(weight_quantization=True, weight_bits=8)
        names = [name for name, _cls in get_pipeline_step_specs(cfg)]
        assert "LIF Affine Fold" in names
        assert names.index("LIF Affine Fold") < names.index("Weight Quantization")
        assert names.index("LIF Affine Fold") > names.index("LIF Adaptation")

    def test_not_selected_when_flag_off(self):
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
        )

        cfg = _config(lif_affine_fold=False)
        names = [name for name, _cls in get_pipeline_step_specs(cfg)]
        assert "LIF Affine Fold" not in names

    def test_fp_novena_schedules_and_passes_the_dag_check(self):
        # [R2c] AQ preconditioning applies to every LIF plan, so the reference
        # entry is promised before the fold even on fp cells; the assembly
        # contract check inside get_pipeline_step_specs must hold.
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
        )

        cfg = _config(activation_quantization=False, firing_mode="Novena")
        names = [name for name, _cls in get_pipeline_step_specs(cfg)]
        assert "LIF Affine Fold" in names
        assert "Activation Quantization" in names
        assert names.index("Activation Quantization") < names.index("LIF Affine Fold")


class TestPremiseSkipByteIdentity:
    """[R2a] a premise-skip is a TRUE no-op: the old step pre-applied the
    half-step entry fold BEFORE the premise check, moving premise-skipped runs
    -0.5..-1.7pp (ledger §2D)."""

    def test_skip_leaves_the_model_byte_identical(self):
        # aq_reference 0.0 makes the premise unsatisfiable (deployed < -SE).
        cfg = _config(lif_half_step_bias=True)
        model = _deployed_model(cfg)
        snapshot = _model_snapshot(model)
        pipeline = _run_step(cfg, model, aq_reference=0.0)
        assert pipeline.cache["LIF Affine Fold.model"] is model
        _assert_model_matches_snapshot(model, snapshot)
        for p in model.get_perceptrons():
            assert not getattr(p, LIF_HALF_STEP_FLAG, False), p.name
            assert not getattr(p, AFFINE_FOLD_FLAG, False), p.name

    def test_skip_reports_the_premise_witness(self):
        # [G7] armed-lever witness: a premise skip must be observable per cell.
        cfg = _config()
        model = _deployed_model(cfg)
        pipeline = _run_step(cfg, model, aq_reference=0.0)
        payload = pipeline.reporter.reports["lif_affine_fold"]
        assert payload["folded"] == 0
        assert payload["skipped_premise"] is True
        assert 0.0 <= payload["deployed_read"] <= 1.0
        assert payload["aq_reference"] == 0.0


class TestTeacherReference:
    """[R2b] the premise fires when the deployed calibration read sits below
    the AQ reference by more than the SE guard, and skips otherwise."""

    def test_premise_fires_below_the_reference_and_folds(self):
        cfg = _config()
        model = _deployed_model(cfg)
        pipeline = _run_step(cfg, model, aq_reference=1.0)
        assert any(
            getattr(p, AFFINE_FOLD_FLAG, False) for p in model.get_perceptrons()
        ), "a random tiny chain reads far below reference 1.0: the fold must fire"
        payload = pipeline.reporter.reports["lif_affine_fold"]
        assert "skipped_premise" not in payload
        assert payload["folded"] >= 1
        assert payload["deployed_read"] < 1.0

    def test_premise_skips_at_or_above_the_reference(self):
        cfg = _config()
        model = _deployed_model(cfg)
        _run_step(cfg, model, aq_reference=0.0)
        assert not any(
            getattr(p, AFFINE_FOLD_FLAG, False) for p in model.get_perceptrons()
        )


class TestProcess:
    def _pin_premise_open(self):
        import pytest

        import mimarsinan.pipelining.pipeline_steps.adaptation.lif_affine_fold_step as sut

        mp = pytest.MonkeyPatch()
        mp.setattr(
            sut, "evaluate_crater_premise",
            lambda *a, **k: CraterPremiseVerdict(
                holds=True, deployed_read=0.5, reference_read=1.0,
                standard_error=0.0,
            ),
        )
        return mp

    def test_folds_and_updates_model(self):
        # The premise arithmetic is contract-tested in test_lif_affine_fold;
        # here it is pinned open so the step's fold path itself is exercised.
        mp = self._pin_premise_open()
        try:
            cfg = _config()
            model = _deployed_model(cfg)
            pipeline = _run_step(cfg, model)
            assert pipeline.cache["LIF Affine Fold.model"] is model
            assert any(
                getattr(p, AFFINE_FOLD_FLAG, False) for p in model.get_perceptrons()
            ), "the step must apply at least one fold on a plain LIF chain"
        finally:
            mp.undo()

    def test_novena_below_s8_is_gated(self):
        cfg = _config(firing_mode="Novena", simulation_steps=4)
        model = _deployed_model(cfg)
        weights = [p.layer.weight.data.clone() for p in model.get_perceptrons()]
        _run_step(cfg, model)
        for p, saved in zip(model.get_perceptrons(), weights):
            assert torch.equal(p.layer.weight.data, saved), (
                "Novena at S<8 must skip the fold (grid overfit, memo §4 C4)"
            )

    def test_half_step_prefold_only_on_the_fold_path(self):
        # [R2a] the estimator fits on the nearest (half-step) chain, so the
        # entry fold lands together with the folds — never on a skip.
        mp = self._pin_premise_open()
        try:
            cfg = _config(lif_half_step_bias=True)
            model = _deployed_model(cfg)
            _run_step(cfg, model)
            non_encoders = [
                p for p in model.get_perceptrons()
                if not getattr(p, "is_encoding_layer", False)
            ]
            assert all(
                getattr(p, LIF_HALF_STEP_FLAG, False) for p in non_encoders
            ), "the affine estimator is fitted on the nearest (half-step) chain"
        finally:
            mp.undo()


class TestFoldRollback:
    """A destructive fold never survives the step (measured 0.94->0.38 collapse)."""

    def test_destructive_fold_is_rolled_back(self):
        import pytest

        import mimarsinan.pipelining.pipeline_steps.adaptation.lif_affine_fold_step as sut

        mp = pytest.MonkeyPatch()
        mp.setattr(
            sut, "evaluate_crater_premise",
            lambda *a, **k: CraterPremiseVerdict(
                holds=True, deployed_read=0.9, reference_read=0.97,
                standard_error=0.005,
            ),
        )

        def _destroy(model, cal_x, steps):
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(0.0)
            return {"folded": 1, "consumer_folds": 1, "readout_folds": 0,
                    "skipped": {}}

        mp.setattr(sut, "apply_lif_affine_fold", _destroy)
        try:
            cfg = _config()
            model = _deployed_model(cfg)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            _run_step(cfg, model)
            after = model.state_dict()
            for k in before:
                assert torch.equal(before[k], after[k]), f"{k} not restored"
        finally:
            mp.undo()
