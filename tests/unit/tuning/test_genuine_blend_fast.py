"""Fast fixed-increment path for the genuine teacher->cascade blend ramp.

With ``ttfs_genuine_blend_ramp=True`` AND ``ttfs_genuine_blend_fast=True`` (cascaded
only) ``TTFSCycleAdaptationTuner`` runs through the ONE orchestrator
(``AdaptationDriver``) with a ``fixed_ladder`` RateScheduler policy instead of the
greedy/bisect search: it walks a FIXED rate schedule (``ttfs_blend_fast_rates``,
default ``[0.5, 0.75, 0.9, 0.97, 1.0]``), training a FIXED number of steps per rate
(``ttfs_blend_fast_steps_per_rate``, default 120) with ONE Adam optimizer + warmup/
cosine LR over the whole schedule, loss ``CE((1-R)*teacher + R*genuine) +
ttfs_genuine_blend_ce_alpha*CE(genuine)`` — mirroring the validated prototype
(``generated/_genuine_ab/full_ramp.py``). It then deploys the PURE genuine cascade
(the shared ``_finalize_run`` tail), commits rate 1.0, and reports a final metric.

The fold (review Rec 2) removed the bespoke ``run()`` engine: the fast path now
INHERITS the DecisionTrace (one ``commit`` record per scheduled rate, where the
bespoke loop dropped the trace entirely) and the finalize/cliff observability through
the same seam every other tuner uses. NO per-cycle rollback / recovery-to-target /
LR-find / stabilization (the ``fixed_ladder`` policy + a fixed-schedule attempt).

These are MECHANISM tests: with the flag ON the fixed_ladder policy runs (EXACTLY
``len(rates) * steps_per_rate`` optimizer steps; one ``commit`` trace record per rate;
a fast marker is set), the committed rate ends at 1.0, the deployed forward is the
pure genuine single-spike cascade, and ``validate()`` returns a metric. With the flag
OFF every other test in the suite (golden traces, the existing blend-ramp +
ttfs-cycle-step tests) must stay byte-identical: the fast flag is opt-in, default-off.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_pipeline(
    tmp_path,
    *,
    schedule="cascaded",
    blend=True,
    fast=True,
    steps_per_rate=4,
    rates=None,
):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg["ttfs_genuine_blend_ramp"] = blend
    cfg["ttfs_genuine_blend_fast"] = fast
    cfg["ttfs_blend_fast_steps_per_rate"] = steps_per_rate
    if rates is not None:
        cfg["ttfs_blend_fast_rates"] = rates
    # Tiny calibration loops keep the unit test fast and deterministic.
    cfg["ttfs_distmatch_bias_iters"] = 3
    cfg["ttfs_distmatch_bias_eta"] = 0.7
    cfg["ttfs_distmatch_quantile"] = 0.99
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_tuner(tmp_path, **kw):
    pipeline = _make_pipeline(tmp_path, **kw)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


# ── Flag ON: the fast fixed-increment path runs ──────────────────────────────


class TestFastPathFlag:
    def test_fast_flag_recognized(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._genuine_blend_fast is True

    def test_fast_flag_requires_blend_ramp(self, tmp_path):
        # fast path is inert without the genuine blend ramp.
        tuner, _, _ = _make_tuner(tmp_path, blend=False, fast=True)
        assert tuner._genuine_blend_ramp is False
        assert tuner._genuine_blend_fast is False

    def test_fast_rates_default_schedule(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True, rates=None)
        assert tuner._blend_fast_rates == [0.5, 0.75, 0.9, 0.97, 1.0]

    def test_fast_rates_accepts_json_list(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True, rates=[0.6, 1.0])
        assert tuner._blend_fast_rates == [0.6, 1.0]


class TestFastRunTakesFastPath:
    def test_run_routes_through_driver_and_records_one_commit_per_rate(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=3)
        tuner.run()
        assert getattr(tuner, "_fast_blend_path", False) is True, (
            "the fast path must mark itself so the fixed_ladder route is provable"
        )
        # Folded into the one orchestrator: the fixed_ladder policy runs through the
        # driver and records ONE commit per scheduled rate — the DecisionTrace the
        # bespoke loop used to drop (the observability win of review Rec 2).
        assert len(tuner._cycle_log) == len(tuner._blend_fast_rates)
        outcomes = [entry["outcome"] for entry in tuner._cycle_log]
        assert outcomes == ["commit"] * len(tuner._blend_fast_rates), (
            "the fixed_ladder policy commits every scheduled rate — no rollback / "
            "catastrophic / recovery cycles"
        )

    def test_run_uses_unified_finalize_tail(self, tmp_path):
        """The fold flows through the shared ``_finalize_run`` (one orchestrator), so
        the standard phase timing — not a bespoke fast-only path — is recorded."""
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        assert "after_run" in tuner._phase_seconds
        assert "fast_blend" in tuner._phase_seconds

    def test_run_runs_exactly_rates_times_steps_optimizer_steps(self, tmp_path):
        torch.manual_seed(0)
        rates = [0.5, 1.0]
        steps = 5
        tuner, _, _ = _make_tuner(
            tmp_path, fast=True, steps_per_rate=steps, rates=rates,
        )
        tuner.run()
        assert tuner._fast_optimizer_steps == len(rates) * steps

    def test_run_commits_rate_one(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        assert tuner._committed_rate == pytest.approx(1.0)

    def test_run_deploys_pure_genuine_cascade(self, tmp_path):
        torch.manual_seed(0)
        tuner, model, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward), (
            "the fast path must run the existing finalize so the deployed forward "
            "is the pure genuine single-spike cascade"
        )
        # Bit-exact to a freshly built cascade on the same weights.
        T = int(tuner.pipeline.config["simulation_steps"])
        x = torch.randn(3, *tuner.pipeline.config["input_shape"])
        fresh = _SegmentSpikeForward(model, T)
        with torch.no_grad():
            torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)

    def test_run_returns_validate_metric(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        result = tuner.run()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # validate() returns the cached final metric.
        validated = tuner.validate()
        assert isinstance(validated, float)
        assert 0.0 <= validated <= 1.0

    def test_run_records_phase_seconds(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        tuner.run()
        assert "fast_blend" in tuner._phase_seconds

    def test_custom_rates_never_invoke_heavy_controller(self, tmp_path):
        """A custom ladder NOT ending in 1.0 is normalized to a trailing 1.0, so the
        ramp finishes through the fixed_ladder fast attempt — never the heavy
        ``_adaptation`` controller (LR-find/recovery/rollback), which would run
        off-recipe after the spanning cosine has decayed to 0."""
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=3, rates=[0.5, 0.9])
        calls = []
        tuner._adaptation = lambda rate: calls.append(rate)  # spy: must never fire
        tuner.run()
        assert calls == [], "the fast path must never invoke the heavy _adaptation controller"
        assert tuner._committed_rate == pytest.approx(1.0)
        assert tuner._fixed_ladder_rates[-1] == pytest.approx(1.0)
        # one fast commit per normalized rate, no extra heavy cycles
        assert len(tuner._cycle_log) == len(tuner._fixed_ladder_rates)

    def test_rerun_resets_fast_scratch(self, tmp_path):
        """Re-running ``run()`` on one instance rebuilds the optimizer + spanning
        cosine and resets the step counter (re-run faithfulness vs the old loop that
        built a fresh optimizer each call); otherwise a re-run re-steps an exhausted
        cosine and double-counts steps."""
        torch.manual_seed(0)
        rates, steps = [0.5, 1.0], 3
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=steps, rates=rates)
        tuner.run()
        first_opt = tuner._fast_optimizer
        assert tuner._fast_optimizer_steps == len(rates) * steps
        tuner.run()
        assert tuner._fast_optimizer_steps == len(rates) * steps  # reset, not doubled
        assert tuner._fast_optimizer is not first_opt  # rebuilt, not the exhausted one

    def test_fast_path_disables_stabilization(self, tmp_path):
        """The fast path trains through the genuine cascade for the whole ramp, so
        the post-finalize stabilization pass is disabled (budget 0) — even if the
        bounded-cosine stabilization flag is also set."""
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        assert tuner._stabilization_budget() == 0


# ── Flag OFF (default): the SmoothAdaptation path is unchanged ────────────────


class TestFastFlagOffUnchanged:
    def test_fast_disabled_by_default(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_genuine_blend_ramp"] = True
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=AdaptationManager(),
        )
        assert tuner._genuine_blend_fast is False

    def test_blend_ramp_without_fast_uses_scheduler(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=False)
        assert tuner._genuine_blend_fast is False
        tuner.run()
        # The SmoothAdaptation scheduler ran: cycles were logged and no fast marker.
        assert getattr(tuner, "_fast_blend_path", False) is False
        assert len(tuner._cycle_log) > 0

    def test_fast_flag_inert_without_blend_ramp(self, tmp_path):
        # blend OFF + fast ON: the value-domain proxy ramp (scheduler) still runs.
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, blend=False, fast=True)
        tuner.run()
        assert getattr(tuner, "_fast_blend_path", False) is False
        assert len(tuner._cycle_log) > 0


# ── Synchronized: the genuine blend (and so the fast path) is ignored ─────────


class TestSynchronizedIgnoresFast:
    def test_synchronized_ignores_fast_flag(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, schedule="synchronized", fast=True)
        assert tuner._genuine_blend_ramp is False
        assert tuner._genuine_blend_fast is False
