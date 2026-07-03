"""Paired McNemar gate: estimator correctness + Monte-Carlo calibration (P2b)."""

import math

import numpy as np
import pytest

from conftest import make_tiny_supermodel
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor


def test_paired_drop_se_exact_counts():
    # 10 examples: 2 discordant ref-right/cand-wrong, 1 ref-wrong/cand-right
    ref = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    cand = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    delta, se = AcceptanceSensor.paired_drop_se(ref, cand)
    assert delta == pytest.approx((2 - 1) / 10)
    assert se == pytest.approx(math.sqrt(2 + 1) / 10)


def test_paired_drop_se_empty_and_mismatch():
    assert AcceptanceSensor.paired_drop_se([], []) == (0.0, 0.0)
    assert AcceptanceSensor.paired_drop_se([1, 0], [1]) == (0.0, 0.0)


def test_paired_gate_fires_only_past_k_se():
    # large asymmetric drop → rollback; symmetric discordance → no rollback
    ref = [1] * 100
    cand_drop = [0] * 20 + [1] * 80          # 20 ref-right→cand-wrong, b01=0
    assert AcceptanceSensor.paired_is_rollback(ref, cand_drop, 2.0) is True
    cand_noise = [1] * 100
    assert AcceptanceSensor.paired_is_rollback(ref, cand_noise, 2.0) is False


def test_min_effect_floors_significant_but_sub_budget_drops():
    # b10=8/400 → delta=0.02, se=sqrt(8)/400≈0.0071; significant (>2·se=0.0141)
    ref = [1] * 400
    cand = [0] * 8 + [1] * 392
    # below a 0.05 budget floor → not rolled back despite significance
    assert AcceptanceSensor.paired_is_rollback(ref, cand, 2.0, min_effect=0.05) is False
    # above the floor (and significant) → rolled back
    assert AcceptanceSensor.paired_is_rollback(ref, cand, 2.0, min_effect=0.0) is True
    # a genuine large drop clears both significance and any small budget
    big = [0] * 60 + [1] * 340
    assert AcceptanceSensor.paired_is_rollback(ref, big, 2.0, min_effect=0.005) is True


def _simulate(rng, n, q10, q01):
    """One paired sample of size n with discordance probs q10/q01."""
    u = rng.random(n)
    ref = np.ones(n, dtype=bool)
    cand = np.ones(n, dtype=bool)
    # discordant: b10 (ref right, cand wrong) then b01 (ref wrong, cand right)
    b10 = u < q10
    b01 = (u >= q10) & (u < q10 + q01)
    cand[b10] = False
    ref[b01] = False
    return ref, cand


def test_montecarlo_false_reject_and_estimator_validity():
    rng = np.random.default_rng(0)
    n, trials, k = 400, 600, 2.0

    # No real drop (q10 == q01), low discordance (a high-accuracy model): the
    # gate should rarely fire and the paired SE should be several-fold tighter.
    deltas, ses, false_rejects = [], [], 0
    for _ in range(trials):
        ref, cand = _simulate(rng, n, q10=0.02, q01=0.02)
        d, se = AcceptanceSensor.paired_drop_se(ref, cand)
        deltas.append(d)
        ses.append(se)
        if AcceptanceSensor.paired_is_rollback(ref, cand, k):
            false_rejects += 1

    false_reject_rate = false_rejects / trials
    assert false_reject_rate < 0.10  # k=2 one-sided tail under the null

    # Estimator validity: mean estimated SE ≈ empirical SD of delta.
    empirical_sd = float(np.std(deltas))
    mean_se = float(np.mean(ses))
    assert mean_se == pytest.approx(empirical_sd, rel=0.25)

    # Paired SE is far tighter than the marginal 0.5/sqrt(N).
    marginal = 0.5 / math.sqrt(n)
    assert mean_se < 0.5 * marginal


def test_montecarlo_true_drop_is_detected():
    rng = np.random.default_rng(1)
    n, trials, k = 400, 400, 2.0
    detected = 0
    for _ in range(trials):
        ref, cand = _simulate(rng, n, q10=0.10, q01=0.02)  # delta_true = 0.08
        if AcceptanceSensor.paired_is_rollback(ref, cand, k):
            detected += 1
    assert detected / trials > 0.9  # a real 8% drop is almost always caught


def _paired_tuner(tmp_path, monkeypatch, cand_correct):
    """A paired-gate tuner whose candidate paired-correctness vector is scripted."""
    import torch

    from conftest import MockPipeline, default_config, override_tuning_policy
    from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
        SmoothAdaptationTuner,
    )

    override_tuning_policy(monkeypatch, use_paired_sensor=True)
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()

    class _T(SmoothAdaptationTuner):
        def _update_and_evaluate(self, rate):
            with torch.no_grad():  # mutate so a rollback restore is observable
                for p in self.model.parameters():
                    p.add_(1.0)
            return 0.85  # non-catastrophic

        def _find_lr(self):
            return 0.001

    tuner = _T(pipeline, model, target_accuracy=0.9, lr=0.001)
    assert tuner._paired_gate is True
    tuner.trainer.train_steps_until_target = lambda *a, **k: None
    tuner.trainer.validate_correctness_on_indices = lambda idx: list(cand_correct)
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9
    tuner._confirm_indices = [0]
    tuner._ref_correct = [True] * 8  # baseline: all correct
    return tuner, model


def test_live_paired_gate_rolls_back_on_real_drop(tmp_path, monkeypatch):
    import torch

    tuner, model = _paired_tuner(tmp_path, monkeypatch, cand_correct=[False] * 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = tuner._adaptation(0.5)
    assert result == 0.0  # paired drop vs fixed baseline → rollback
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)
    tuner.close()


def test_live_paired_gate_commits_when_no_drop(tmp_path, monkeypatch):
    tuner, _ = _paired_tuner(tmp_path, monkeypatch, cand_correct=[True] * 8)
    result = tuner._adaptation(0.5)
    assert result == 0.5  # candidate matches baseline → commit
    tuner.close()


def test_paired_gate_uses_single_post_eval_pass(tmp_path, monkeypatch):
    """D6: under the paired gate the post-recovery accuracy is derived from the
    SAME correctness vector the rollback gate uses — no separate marginal
    validate_n_batches pass (the redundant second eval the +139% cost included)."""
    tuner, _ = _paired_tuner(tmp_path, monkeypatch, cand_correct=[True] * 8)
    tuner._last_post_acc = 0.9  # skip the pre-cycle validate_n_batches as well

    counts = {"vn": 0, "vc": 0}
    real_vc = tuner.trainer.validate_correctness_on_indices

    def counted_vn(n):
        counts["vn"] += 1
        return 0.9

    def counted_vc(idx):
        counts["vc"] += 1
        return real_vc(idx)

    tuner.trainer.validate_n_batches = counted_vn
    tuner.trainer.validate_correctness_on_indices = counted_vc

    result = tuner._adaptation(0.5)
    assert result == 0.5  # committed (no drop)
    assert counts["vc"] == 1  # one paired correctness pass (decision + post_acc)
    assert counts["vn"] == 0  # the redundant marginal post pass is gone
    tuner.close()


def test_correctness_primitive_reads_validation_cache(tmp_path):
    from conftest import MockPipeline, default_config
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
    from mimarsinan.model_training.basic_trainer import BasicTrainer

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    trainer = BasicTrainer(
        model, cfg["device"],
        DataLoaderFactory(pipeline.data_provider_factory),
        pipeline.loss,
    )
    # test-set isolation: the primitive must never touch the test loader
    trainer.test = lambda *a, **k: (_ for _ in ()).throw(AssertionError("test() called"))

    c0 = trainer.validate_correctness_on_indices([0, 1])
    c1 = trainer.validate_correctness_on_indices([0, 1])
    assert isinstance(c0, list) and all(isinstance(v, bool) for v in c0)
    assert c0 == c1  # same examples each call → paired/deterministic
    trainer.close()


# ── D4: global_budget floor — consistency + validation (not the 0.005 footgun) ──
# The ablation (docs/tuning_optimization_flags.md §1) measured a 0.005 floor to
# ERASE the paired accuracy gain; 0.0 (no §8.2 floor) is strictly better and is
# the TuningPolicy default. These tests pin that contract: 0.0 stays the default,
# the cycle mixin reads it (no silent 0.005 divergence), 0.0 is a valid paired
# setting, and only a *negative* budget is rejected.

def _budget_tuner(tmp_path, monkeypatch, *, paired, global_budget=None):
    from conftest import MockPipeline, default_config, override_tuning_policy
    from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
        SmoothAdaptationTuner,
    )

    overrides = {}
    if paired:
        overrides["use_paired_sensor"] = True
    if global_budget is not None:
        overrides["global_budget"] = global_budget
    if overrides:
        override_tuning_policy(monkeypatch, **overrides)
    pipeline = MockPipeline(config=default_config(), working_directory=str(tmp_path))

    class _T(SmoothAdaptationTuner):
        def _update_and_evaluate(self, rate):
            return 0.9

        def _find_lr(self):
            return 0.001

    tuner = _T(pipeline, make_tiny_supermodel(), target_accuracy=0.9, lr=0.001)
    return tuner


def test_global_budget_policy_default_is_zero():
    from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

    assert TUNING_POLICY.global_budget == 0.0


def test_cycle_global_budget_matches_policy_default(tmp_path, monkeypatch):
    # default policy → the mixin reads 0.0 (the SSOT default), not a divergent
    # hardcoded 0.005.
    tuner = _budget_tuner(tmp_path, monkeypatch, paired=False)
    assert tuner._global_budget == 0.0
    tuner.close()


def test_paired_with_zero_budget_is_allowed(tmp_path, monkeypatch):
    # 0.0 (no floor) is the measured best-accuracy operating point — it MUST stay
    # expressible with the paired gate on, not raised as a misconfiguration.
    tuner = _budget_tuner(tmp_path, monkeypatch, paired=True, global_budget=0.0)
    assert tuner._paired_gate is True and tuner._global_budget == 0.0
    tuner.close()


def test_negative_global_budget_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="global_budget"):
        _budget_tuner(tmp_path, monkeypatch, paired=True, global_budget=-0.01)
