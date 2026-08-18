"""TS5: the adaptation ledger — one event per controller branch the run took.

The ledger reports CODE REALITY: the branch taxonomy here is read off the
shipped controller (``AdaptationDriver.run_cycle``, ``RateScheduler.run``,
``SmoothAdaptationCycleMixin._commit_cycle``, ``experimental_walk_recovery``,
``frontier/endpoint_recovery``), never invented. Nothing in this file may change
a tuning decision — the ledger is telemetry that a run seals.

The completeness pins below are the mutation targets: dropping any single
emission must turn its pin red.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_scripted_run_tuner,
    make_tiny_supermodel,
    override_tuning_policy,
)

from mimarsinan.tuning.orchestration import adaptation_ledger as led
from mimarsinan.tuning.orchestration.acceptance_sensor import (
    PROBE_INSTANT,
    PROBE_MARGINAL,
    PROBE_PAIRED,
)
from mimarsinan.tuning.orchestration.adaptation_driver import (
    AdaptationDriver,
    CycleContext,
)
from mimarsinan.tuning.orchestration.adaptation_ledger import (
    AdaptationLedger,
    Endpoint,
    Escalation,
    Proposal,
    Recovery,
    Refinement,
    Verdict,
)
from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
)

EPS = 2 ** -6


# --------------------------------------------------------------------------- #
# The accumulator: totals are sums over the events, nothing else.
# --------------------------------------------------------------------------- #


def _populated() -> AdaptationLedger:
    ledger = AdaptationLedger()
    ledger.propose(rate=0.5, committed=0.0)
    ledger.recover(kind=led.RECOVERY_TRAINED, steps=40, lr=1e-3)
    ledger.verdict(
        outcome=led.COMMIT, probe_metric=PROBE_MARGINAL, reading=0.9,
        threshold=0.85, probe_reads=3,
    )
    ledger.propose(rate=1.0, committed=0.5)
    ledger.recover(kind=led.RECOVERY_LR_REFUSED, steps=0, lr=0.0)
    ledger.verdict(
        outcome=led.ROLLBACK, probe_metric=PROBE_MARGINAL, reading=0.5,
        threshold=0.85, probe_reads=2,
    )
    ledger.refine(led.BISECT_STEP, "0.500000 -> 0.250000")
    ledger.propose(rate=0.75, committed=0.5)
    ledger.recover(kind=led.RECOVERY_TRAINED, steps=10, lr=1e-3)
    ledger.verdict(
        outcome=led.CATASTROPHIC, probe_metric=PROBE_INSTANT, reading=0.1,
        threshold=None, probe_reads=1,
    )
    ledger.escalate(led.EPSILON_FLOOR, "step 0.007812 < eps 0.015625")
    ledger.endpoint(
        stage="ActivationQuantizationTuner", steps=200, budget_steps=600,
        engaged=True, armed=True, reached=False,
    )
    ledger.complete(led.EPSILON_FLOOR)
    return ledger


class TestAccumulator:
    def test_totals_are_sums_over_the_events(self):
        totals = _populated().totals()
        assert totals == {
            "proposed": 3,
            "accepted": 1,
            "rejected": 2,
            "retries": 1,
            "recovery_steps": 50,
            "probe_evals": 6,
            "endpoint_steps": 200,
            "total_steps": 250,
        }

    def test_totals_match_the_event_lists_they_summarize(self):
        ledger = _populated()
        totals = ledger.totals()
        assert totals["proposed"] == len(ledger.proposals)
        assert totals["accepted"] + totals["rejected"] == len(ledger.verdicts)
        assert totals["retries"] == sum(
            1 for r in ledger.refinements if r.kind == led.BISECT_STEP
        )
        assert totals["recovery_steps"] == sum(r.steps for r in ledger.recoveries)
        assert totals["probe_evals"] == sum(v.probe_reads for v in ledger.verdicts)
        assert totals["endpoint_steps"] == sum(e.steps for e in ledger.endpoints)
        assert totals["total_steps"] == (
            totals["recovery_steps"] + totals["endpoint_steps"]
        )

    def test_empty_ledger_totals_are_all_zero(self):
        assert set(AdaptationLedger().totals().values()) == {0}

    def test_stalls_by_path_counts_escalations_per_path(self):
        ledger = AdaptationLedger()
        ledger.escalate(led.EPSILON_FLOOR, "")
        ledger.escalate(led.EPSILON_FLOOR, "")
        ledger.escalate(led.ROUND_BUDGET, "")
        assert ledger.stalls_by_path() == {led.EPSILON_FLOOR: 2, led.ROUND_BUDGET: 1}

    def test_cycle_index_follows_the_proposal_count(self):
        ledger = AdaptationLedger()
        ledger.propose(rate=0.5, committed=0.0)
        ledger.verdict(
            outcome=led.COMMIT, probe_metric=PROBE_MARGINAL, reading=0.9,
            threshold=0.8, probe_reads=2,
        )
        ledger.propose(rate=1.0, committed=0.5)
        ledger.recover(kind=led.RECOVERY_TRAINED, steps=1, lr=1e-3)
        assert [p.cycle_index for p in ledger.proposals] == [0, 1]
        assert ledger.verdicts[0].cycle_index == 0
        assert ledger.recoveries[0].cycle_index == 1

    def test_events_are_frozen(self):
        ledger = _populated()
        for event in (
            ledger.proposals[0], ledger.verdicts[0], ledger.refinements[0],
            ledger.escalations[0], ledger.recoveries[0], ledger.endpoints[0],
        ):
            field = dataclasses.fields(event)[0].name
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(event, field, 99)

    def test_to_dict_round_trips_through_json(self):
        payload = _populated().to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_to_dict_carries_every_event_group_and_the_summary(self):
        payload = _populated().to_dict()
        assert set(payload) == {
            "proposals", "verdicts", "refinements", "escalations",
            "recoveries", "endpoints", "totals", "stalls_by_path",
            "completed_via",
        }
        assert len(payload["proposals"]) == 3
        assert len(payload["verdicts"]) == 3
        assert len(payload["refinements"]) == 1
        assert len(payload["escalations"]) == 1
        assert len(payload["recoveries"]) == 3
        assert len(payload["endpoints"]) == 1


class TestCompletedVia:
    def test_the_artifact_answers_which_path_completed_the_run(self):
        # The Fig-7.2 question, answered mechanically from the sealed payload.
        payload = _populated().to_dict()
        assert payload["completed_via"] == led.EPSILON_FLOOR

    def test_a_run_that_never_completed_seals_none(self):
        assert AdaptationLedger().to_dict()["completed_via"] is None

    def test_completion_is_the_last_path_the_search_exited_through(self):
        ledger = AdaptationLedger()
        ledger.complete(led.ROUND_BUDGET)
        ledger.complete(led.REACHED_FULL_RATE)
        assert ledger.completed_via == led.REACHED_FULL_RATE


# --------------------------------------------------------------------------- #
# The driver's cycle branches — a scripted host drives the REAL run_cycle.
# --------------------------------------------------------------------------- #


class _ScriptedHost:
    """A controller host that scripts one cycle's verdict for the real driver.

    Implements exactly the phase surface ``AdaptationDriver.run_cycle`` drives.
    """

    def __init__(
        self, ledger, *, outcome="commit", paired=False, pre_probe_drawn=True,
        recovery_steps=0, lr=1e-3, committed=0.25,
    ):
        self._adaptation_ledger = ledger
        self._committed_rate = committed
        self._gradual_train_steps = 0
        self._outcome = outcome
        self._paired = paired
        self._pre_probe_drawn = pre_probe_drawn
        self._recovery_steps = recovery_steps
        self._lr = lr
        self.calls = []

    def _begin_cycle(self, rate):
        self.calls.append("begin")
        return CycleContext(
            rate=float(rate),
            t_cycle_start=0.0,
            pre_cycle_acc=0.9,
            pre_probe_drawn=self._pre_probe_drawn,
        )

    def _probe_instant(self, ctx):
        self.calls.append("probe")
        ctx.instant_acc = 0.1 if self._outcome == "catastrophic" else 0.85
        ctx.is_catastrophic = self._outcome == "catastrophic"

    def _recover(self, ctx):
        self.calls.append("recover")
        ctx.lr = self._lr
        self._gradual_train_steps += self._recovery_steps

    def _measure_post(self, ctx):
        self.calls.append("measure")
        ctx.rollback_threshold = 0.88
        ctx.post_acc = 0.5 if self._outcome == "rollback" else 0.95
        ctx.rolled_back = self._outcome == "rollback"
        if self._paired:
            ctx.cand_correct = [True, False]

    def _rollback_cycle(self, ctx, outcome):
        self.calls.append(("rollback", outcome))
        return self._committed_rate

    def _commit_cycle(self, ctx):
        self.calls.append("commit")
        self._committed_rate = ctx.rate
        return ctx.rate


def _drive(**kwargs):
    ledger = AdaptationLedger()
    host = _ScriptedHost(ledger, **kwargs)
    AdaptationDriver.run_cycle(host, 0.5)
    return ledger, host


class TestCycleBranchEmission:
    def test_commit_branch_emits_one_proposal_one_recovery_one_verdict(self):
        ledger, _ = _drive(outcome="commit", recovery_steps=40)
        assert len(ledger.proposals) == 1
        assert len(ledger.recoveries) == 1
        assert len(ledger.verdicts) == 1
        assert ledger.verdicts[0].outcome == led.COMMIT
        assert ledger.verdicts[0].accepted is True
        assert ledger.verdicts[0].rolled_back is False

    def test_the_proposal_carries_the_increment_the_controller_asked_for(self):
        ledger, _ = _drive(committed=0.25)
        proposal = ledger.proposals[0]
        assert proposal.rate == pytest.approx(0.5)
        assert proposal.committed == pytest.approx(0.25)
        assert proposal.increment == pytest.approx(0.25)

    def test_rollback_branch_emits_a_rejecting_verdict_that_rolled_back(self):
        ledger, _ = _drive(outcome="rollback")
        assert len(ledger.verdicts) == 1
        verdict = ledger.verdicts[0]
        assert verdict.outcome == led.ROLLBACK
        assert verdict.accepted is False
        assert verdict.rolled_back is True
        assert verdict.reading == pytest.approx(0.5)
        assert verdict.threshold == pytest.approx(0.88)

    def test_catastrophic_branch_emits_one_verdict_and_no_recovery(self):
        # The real branch short-circuits BEFORE _recover: no recovery leg ran,
        # so the ledger must not invent one.
        ledger, host = _drive(outcome="catastrophic")
        assert host.calls == ["begin", "probe", ("rollback", "catastrophic")]
        assert len(ledger.verdicts) == 1
        assert ledger.recoveries == []
        verdict = ledger.verdicts[0]
        assert verdict.outcome == led.CATASTROPHIC
        assert verdict.accepted is False
        assert verdict.rolled_back is True
        assert verdict.threshold is None

    def test_verdict_names_the_sensor_basis_it_read(self):
        assert _drive(outcome="commit")[0].verdicts[0].probe_metric == PROBE_MARGINAL
        assert _drive(outcome="commit", paired=True)[0].verdicts[0].probe_metric == (
            PROBE_PAIRED
        )
        assert _drive(outcome="catastrophic")[0].verdicts[0].probe_metric == (
            PROBE_INSTANT
        )

    def test_probe_reads_count_the_readings_the_gate_consumed(self):
        # pre-cycle probe (only when no post accuracy was carried) + instant +
        # post (skipped on the catastrophic short-circuit).
        assert _drive(outcome="commit")[0].verdicts[0].probe_reads == 3
        assert _drive(outcome="commit", pre_probe_drawn=False)[0].verdicts[
            0
        ].probe_reads == 2
        assert _drive(outcome="catastrophic")[0].verdicts[0].probe_reads == 2
        assert _drive(outcome="catastrophic", pre_probe_drawn=False)[0].verdicts[
            0
        ].probe_reads == 1

    def test_recovery_records_the_steps_the_cycle_consumed(self):
        ledger, _ = _drive(outcome="commit", recovery_steps=40)
        recovery = ledger.recoveries[0]
        assert recovery.kind == led.RECOVERY_TRAINED
        assert recovery.steps == 40
        assert recovery.lr == pytest.approx(1e-3)

    def test_a_refused_lr_sweep_records_a_zero_step_recovery(self):
        # [LR-REFUSE] the cycle records lr=0.0 and trains nothing.
        ledger, _ = _drive(outcome="commit", lr=0.0, recovery_steps=0)
        recovery = ledger.recoveries[0]
        assert recovery.kind == led.RECOVERY_LR_REFUSED
        assert recovery.steps == 0

    def test_a_host_without_a_ledger_is_untouched(self):
        # Byte-identical no-fragment A/B: the driver must not require telemetry.
        class _Bare(_ScriptedHost):
            def __init__(self):
                super().__init__(AdaptationLedger())
                del self._adaptation_ledger

        host = _Bare()
        assert AdaptationDriver.run_cycle(host, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# The scheduler's refinement + escalation branches (the REAL RateScheduler).
# --------------------------------------------------------------------------- #


def _cliff(alpha_star):
    state = {"committed": 0.0}

    def attempt(target):
        if target <= alpha_star + 1e-12:
            state["committed"] = target
        return state["committed"]

    return attempt


class TestSchedulerBranchEmission:
    def test_a_smooth_run_completes_by_reaching_full_rate(self):
        ledger = AdaptationLedger()
        RateScheduler(epsilon=EPS, ledger=ledger).run(0.0, lambda t: t)
        assert ledger.completed_via == led.REACHED_FULL_RATE
        assert ledger.escalations == []
        assert ledger.refinements == []

    def test_each_bisection_halving_records_one_retry_refinement(self):
        ledger = AdaptationLedger()
        RateScheduler(epsilon=EPS, ledger=ledger).run(0.0, _cliff(0.5))
        bisects = [r for r in ledger.refinements if r.kind == led.BISECT_STEP]
        assert bisects, "the bisection refinement must be recorded"
        assert ledger.totals()["retries"] == len(bisects)
        assert "->" in bisects[0].detail

    def test_bisection_underflow_escalates_and_completes_via_the_epsilon_floor(self):
        ledger = AdaptationLedger()
        RateScheduler(epsilon=EPS, ledger=ledger).run(0.0, _cliff(0.0))
        assert ledger.stalls_by_path() == {led.EPSILON_FLOOR: 1}
        assert ledger.completed_via == led.EPSILON_FLOOR

    def test_the_round_budget_escalates_and_completes_the_run(self):
        ledger = AdaptationLedger()
        RateScheduler(
            epsilon=EPS, policy="uniform_ladder", initial_step=0.25, max_rounds=2,
            ledger=ledger,
        ).run(0.0, lambda t: t)
        assert ledger.stalls_by_path() == {led.ROUND_BUDGET: 1}
        assert ledger.completed_via == led.ROUND_BUDGET

    def test_a_refused_one_shot_escalates_without_bisecting(self):
        ledger = AdaptationLedger()
        RateScheduler(epsilon=EPS, policy="one_shot_only", ledger=ledger).run(
            0.0, _cliff(0.5)
        )
        assert ledger.stalls_by_path() == {led.ONE_SHOT_REFUSED: 1}
        assert ledger.completed_via == led.ONE_SHOT_REFUSED
        assert ledger.refinements == []

    def test_the_fixed_ladder_completes_when_its_rate_list_is_exhausted(self):
        ledger = AdaptationLedger()
        RateScheduler(
            epsilon=EPS, policy="fixed_ladder", rates=[0.5, 1.0], ledger=ledger,
        ).run(0.0, lambda t: t)
        assert ledger.completed_via == led.FIXED_LADDER_EXHAUSTED

    def test_a_scheduler_without_a_ledger_behaves_identically(self):
        with_ledger = RateScheduler(epsilon=EPS, ledger=AdaptationLedger()).run(
            0.0, _cliff(0.5)
        )
        without = RateScheduler(epsilon=EPS).run(0.0, _cliff(0.5))
        assert with_ledger == pytest.approx(without)


# --------------------------------------------------------------------------- #
# The driver's lossless-entry fast path.
# --------------------------------------------------------------------------- #


class TestLosslessEntryEscalation:
    def test_the_short_circuit_escalates_and_completes_the_run(self):
        ledger = AdaptationLedger()
        driver = AdaptationDriver(
            scheduler=None,
            attempt=lambda rate: rate,
            finalize=lambda: "finalized",
            entry_short_circuit=lambda: True,
            ledger=ledger,
        )
        assert driver.run() == "finalized"
        assert ledger.stalls_by_path() == {led.LOSSLESS_ENTRY: 1}
        assert ledger.completed_via == led.LOSSLESS_ENTRY


# --------------------------------------------------------------------------- #
# The cycle mixin's own refinements — the REAL _commit_cycle branches.
# --------------------------------------------------------------------------- #


class _CycleTuner(SmoothAdaptationTuner):
    """Drives ``_adaptation`` with a scripted ``[pre, post]`` validate sequence."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._validate_seq = []
        self._idx = 0
        self._adaptation_ledger = AdaptationLedger()
        self.trainer.train_steps_until_target = lambda *a, **k: None

    def _update_and_evaluate(self, rate):
        return 0.85

    def _find_lr(self):
        return 0.001

    def _validate_n(self, _n):
        i = self._idx
        self._idx += 1
        seq = self._validate_seq
        return seq[i] if i < len(seq) else seq[-1]

    def drive(self, rate, validate_seq):
        self._validate_seq = list(validate_seq)
        self._idx = 0
        self._last_post_acc = None
        self.trainer.validate_n_batches = self._validate_n
        return self._adaptation(rate)


def _cycle_tuner(tmp_path, monkeypatch=None, **policy_overrides):
    if policy_overrides:
        override_tuning_policy(monkeypatch, **policy_overrides)
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = _CycleTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001)
    tuner._rollback_tolerance = 0.02
    tuner._validation_baseline = 0.5
    tuner.target_adjuster.target_metric = 0.99
    tuner.target_adjuster.original_metric = 0.99
    tuner.target_adjuster.floor = 0.1
    return tuner


class TestCommitCycleRefinements:
    def test_a_missed_target_streak_records_one_target_relaxation(
        self, tmp_path, deterministic_rng
    ):
        # Three consecutive commits that miss the target relax it exactly once
        # (_STUCK_STREAK_REQUIRED), and the relaxation invalidates the LR cache.
        tuner = _cycle_tuner(tmp_path)
        for rate in (0.2, 0.4, 0.6):
            tuner.drive(rate, validate_seq=[0.90, 0.90])
        kinds = [r.kind for r in tuner._adaptation_ledger.refinements]
        assert kinds.count(led.TARGET_RELAXATION) == 1
        assert kinds.count(led.LR_REFIND) == 1
        tuner.close()

    def test_refind_on_miss_records_one_lr_refind_per_missed_commit(
        self, tmp_path, deterministic_rng, monkeypatch
    ):
        tuner = _cycle_tuner(tmp_path, monkeypatch, refind_lr_on_miss=True)
        for rate in (0.2, 0.4):
            tuner.drive(rate, validate_seq=[0.90, 0.90])
        kinds = [r.kind for r in tuner._adaptation_ledger.refinements]
        assert kinds == [led.LR_REFIND, led.LR_REFIND]
        tuner.close()

    def test_a_reached_target_records_no_refinement(
        self, tmp_path, deterministic_rng
    ):
        tuner = _cycle_tuner(tmp_path)
        tuner.target_adjuster.target_metric = 0.5  # comfortably reached
        tuner.drive(0.2, validate_seq=[0.90, 0.90])
        assert tuner._adaptation_ledger.refinements == []
        tuner.close()


# --------------------------------------------------------------------------- #
# A whole real run: the ledger the tuner seals is internally consistent.
# --------------------------------------------------------------------------- #


def _run_tuner(tmp_path, instant_fn, post_fn, ladder=False):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = make_scripted_run_tuner(
        pipeline, make_tiny_supermodel(),
        instant_fn=instant_fn, post_fn=post_fn, ladder=ladder,
    )
    tuner.run()
    return tuner


class TestWholeRun:
    def test_a_smooth_run_seals_a_consistent_ledger(self, tmp_path, deterministic_rng):
        tuner = _run_tuner(tmp_path, lambda r: 0.87, lambda r: 0.9)
        ledger = tuner._adaptation_ledger
        totals = ledger.totals()
        assert totals["proposed"] == len(ledger.proposals) >= 1
        assert totals["accepted"] + totals["rejected"] == len(ledger.verdicts)
        assert totals["proposed"] == len(ledger.verdicts)
        assert ledger.completed_via == led.REACHED_FULL_RATE
        tuner.close()

    def test_a_cliff_run_records_rejections_and_bisection_retries(
        self, tmp_path, deterministic_rng
    ):
        # The one-shot at 1.0 collapses; the driver bisects back down.
        instant = lambda r: 0.1 if r >= 0.99 else 0.85
        tuner = _run_tuner(tmp_path, instant, lambda r: 0.9)
        ledger = tuner._adaptation_ledger
        assert ledger.totals()["rejected"] >= 1
        assert ledger.totals()["retries"] >= 1
        assert any(v.outcome == led.CATASTROPHIC for v in ledger.verdicts)
        tuner.close()

    def test_a_run_forced_to_full_rate_escalates(self, tmp_path, deterministic_rng):
        # Nothing above the committed rate is ever feasible, so the natural ramp
        # stalls and _after_run forces the rate to 1.0 — a real escalation.
        instant = lambda r: 0.1
        tuner = _run_tuner(tmp_path, instant, lambda r: 0.9)
        assert led.FORCED_FULL_RATE in tuner._adaptation_ledger.stalls_by_path()
        tuner.close()

    def test_the_ledger_json_serializes_after_a_real_run(
        self, tmp_path, deterministic_rng
    ):
        tuner = _run_tuner(tmp_path, lambda r: 0.87, lambda r: 0.9, ladder=True)
        payload = tuner._adaptation_ledger.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["completed_via"] is not None
        tuner.close()

    def test_a_fresh_run_starts_from_an_empty_ledger(self, tmp_path, deterministic_rng):
        tuner = _run_tuner(tmp_path, lambda r: 0.87, lambda r: 0.9)
        first = tuner._adaptation_ledger.totals()["proposed"]
        tuner.run()
        assert tuner._adaptation_ledger.totals()["proposed"] == first
        tuner.close()


# --------------------------------------------------------------------------- #
# The experimental walk-recovery escalations (default-off knob).
# --------------------------------------------------------------------------- #


class _WalkTuner:
    def __init__(self, config):
        self.pipeline = type("P", (), {"config": config})()
        self.model = object()
        self._adaptation_ledger = AdaptationLedger()
        self.installed = None

    def _install_forward(self, forward):
        self.installed = forward


class TestWalkRecoveryEscalations:
    def _armed_config(self):
        return {
            "lif_exact_qat_walk_recovery": True,
            "spiking_family": "lif",
            "lif_exact_qat": True,
            "simulation_steps": 8,
        }

    def test_installing_the_deployed_walk_records_one_escalation(self, monkeypatch):
        from mimarsinan.tuning.orchestration import experimental_walk_recovery as ewr

        monkeypatch.setattr(ewr, "walk_recovery_armed", lambda cfg: True)
        monkeypatch.setattr(
            ewr, "exact_qat_training_forward", lambda model, cfg: "forward"
        )
        import mimarsinan.spiking.segment_partition as sp

        monkeypatch.setattr(sp, "graph_has_host_compute_ops", lambda m: True)
        tuner = _WalkTuner(self._armed_config())
        assert ewr.install_walk_for_aq_recovery(tuner) is True
        assert tuner._adaptation_ledger.stalls_by_path() == {
            led.WALK_RECOVERY_INSTALL: 1
        }

    def test_a_disarmed_knob_records_nothing(self, monkeypatch):
        from mimarsinan.tuning.orchestration import experimental_walk_recovery as ewr

        monkeypatch.setattr(ewr, "walk_recovery_armed", lambda cfg: False)
        tuner = _WalkTuner({})
        assert ewr.install_walk_for_aq_recovery(tuner) is False
        assert tuner._adaptation_ledger.escalations == []

    def test_restoring_the_lif_recovery_budget_records_one_escalation(
        self, monkeypatch
    ):
        from mimarsinan.tuning.orchestration import experimental_walk_recovery as ewr

        monkeypatch.setattr(ewr, "walk_recovery_armed", lambda cfg: True)
        import mimarsinan.spiking.segment_partition as sp

        monkeypatch.setattr(sp, "graph_has_host_compute_ops", lambda m: True)

        class _Plan:
            exact_qat = True

            def restore_recovery_for_host_graph(self, config):
                return "restored"

        tuner = _WalkTuner(self._armed_config())
        assert ewr.restore_lif_recovery_budget(_Plan(), tuner) == "restored"
        assert tuner._adaptation_ledger.stalls_by_path() == {
            led.LIF_RECOVERY_BUDGET_RESTORE: 1
        }


# --------------------------------------------------------------------------- #
# The endpoint stage's step consumption.
# --------------------------------------------------------------------------- #


class TestEndpointEmission:
    def test_the_endpoint_event_carries_the_steps_the_stage_consumed(self):
        ledger = AdaptationLedger()
        led.record_endpoint(
            ledger,
            stage="ActivationQuantizationTuner",
            steps=120,
            budget_steps=600,
            engaged=True,
            armed=True,
            reached=False,
            rolled_back=False,
            divergence_rescued=False,
            decoupled=False,
        )
        assert ledger.totals()["endpoint_steps"] == 120
        assert ledger.totals()["total_steps"] == 120
        assert ledger.endpoints[0].stage == "ActivationQuantizationTuner"
        assert ledger.escalations == []

    def test_each_fired_endpoint_guard_records_its_own_escalation(self):
        ledger = AdaptationLedger()
        led.record_endpoint(
            ledger, stage="T", steps=10, budget_steps=10, engaged=True, armed=True,
            reached=False, rolled_back=True, divergence_rescued=True, decoupled=True,
        )
        assert ledger.stalls_by_path() == {
            led.ENDPOINT_ROLLBACK: 1,
            led.ENDPOINT_DIVERGENCE_RESCUE: 1,
            led.ENDPOINT_DECOUPLED: 1,
        }

    def test_a_tuner_without_a_ledger_is_a_no_op(self):
        led.record_endpoint(
            None, stage="T", steps=10, budget_steps=10, engaged=True, armed=True,
            reached=True, rolled_back=False, divergence_rescued=False,
            decoupled=False,
        )  # must not raise


def _endpoint_tuner(tmp_path):
    """A LIF tuner scaffolded exactly as the endpoint stage expects to find it."""
    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg.update({
        "spiking_mode": "lif", "firing_mode": "Default", "thresholding_mode": "<",
        "simulation_steps": 4, "lif_blend_fast": True,
        "lif_blend_fast_steps_per_rate": 2, "lif_blend_fast_rates": [0.5, 1.0],
        "endpoint_recovery_steps": 0,
    })
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    tuner = LIFAdaptationTuner(
        pipeline, model=make_tiny_supermodel(), target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )
    tuner._phase_seconds = {}
    tuner._mbh_rung_index = -1
    tuner._mbh_gate_state = None
    tuner._rollback_tolerance = 0.0
    tuner._fast_optimizer_steps = 0
    tuner._adaptation_ledger = AdaptationLedger()
    return tuner


class TestEndpointStageEmission:
    """The REAL P1'' stage records its consumption on the run's ledger."""

    def _patch_engine(self, monkeypatch, reads, steps_used=7):
        from mimarsinan.tuning.orchestration.frontier import endpoint_recovery
        from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine

        stream = iter(reads)
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(stream),
        )
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target",
            staticmethod(lambda *a, **k: (0.5, steps_used)),
        )
        return endpoint_recovery

    def test_an_engaged_stage_records_one_endpoint_event(
        self, tmp_path, deterministic_rng, monkeypatch
    ):
        from mimarsinan.tuning.orchestration import dhat_highwater

        tuner = _endpoint_tuner(tmp_path)
        try:
            module = self._patch_engine(monkeypatch, [0.2, 0.6])
            dhat_highwater.observe(tuner.pipeline, 0.9)
            report = module.run_endpoint_recovery(tuner, base_steps=300)
            ledger = tuner._adaptation_ledger
            assert len(ledger.endpoints) == 1
            event = ledger.endpoints[0]
            assert event.stage == "LIFAdaptationTuner"
            assert event.steps == report.steps_used == 7
            assert event.engaged is True
            assert ledger.totals()["endpoint_steps"] == 7
            assert ledger.escalations == []
        finally:
            tuner.close()

    def test_a_stage_that_ends_below_entry_escalates_a_rollback(
        self, tmp_path, deterministic_rng, monkeypatch
    ):
        from mimarsinan.tuning.orchestration import dhat_highwater

        tuner = _endpoint_tuner(tmp_path)
        try:
            tuner._rollback_tolerance = 0.01
            module = self._patch_engine(monkeypatch, [0.5, 0.1])
            dhat_highwater.observe(tuner.pipeline, 0.9)
            module.run_endpoint_recovery(tuner, base_steps=300)
            ledger = tuner._adaptation_ledger
            assert len(ledger.endpoints) == 1
            assert ledger.stalls_by_path() == {led.ENDPOINT_ROLLBACK: 1}
        finally:
            tuner.close()


# --------------------------------------------------------------------------- #
# The step artifact: <Step>.adaptation_ledger.json, sealed via add_entry.
# --------------------------------------------------------------------------- #


def _run_clamp_step(mock_pipeline):
    from mimarsinan.pipelining.pipeline_steps.adaptation.clamp_adaptation_step import (
        ClampAdaptationStep,
    )
    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

    from conftest import make_activation_scale_stats

    model = make_tiny_supermodel()
    scales = [1.0] * len(model.get_perceptrons())
    mock_pipeline.config["activation_quantization"] = True
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline._target_metric = 0.5
    mock_pipeline.seed("model", model, step_name="Activation Adaptation")
    mock_pipeline.seed(
        "adaptation_manager", AdaptationManager(), step_name="Activation Adaptation"
    )
    mock_pipeline.seed("activation_scales", scales, step_name="Activation Analysis")
    mock_pipeline.seed(
        "activation_scale_stats",
        make_activation_scale_stats(model, scales, num_batches=2),
        step_name="Activation Analysis",
    )
    step = ClampAdaptationStep(mock_pipeline)
    step.name = "Clamp Adaptation"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


class TestStepArtifact:
    def test_every_tuner_step_promises_the_ledger_entry(self):
        from mimarsinan.pipelining.core.steps.tuner_pipeline_step import (
            TunerPipelineStep,
        )
        from mimarsinan.pipelining.pipeline_steps.quantization.activation_quantization_step import (  # noqa: E501
            ActivationQuantizationStep,
        )

        assert led.LEDGER_ENTRY_KEY in TunerPipelineStep.PROMISES
        # A subclass that adds its own promise must not drop the base's.
        assert led.LEDGER_ENTRY_KEY in ActivationQuantizationStep.PROMISES

    def test_a_tuner_step_seals_its_ledger_as_a_cache_entry(
        self, mock_pipeline, deterministic_rng
    ):
        step = _run_clamp_step(mock_pipeline)
        payload = mock_pipeline.cache[f"Clamp Adaptation.{led.LEDGER_ENTRY_KEY}"]
        assert json.loads(json.dumps(payload)) == payload
        assert payload["totals"]["proposed"] == len(payload["proposals"])
        assert payload["totals"]["proposed"] >= 1

    def test_the_sealed_artifact_answers_which_path_completed_the_run(
        self, mock_pipeline, deterministic_rng
    ):
        step = _run_clamp_step(mock_pipeline)
        payload = mock_pipeline.cache[f"Clamp Adaptation.{led.LEDGER_ENTRY_KEY}"]
        assert payload["completed_via"] in {
            led.REACHED_FULL_RATE, led.EPSILON_FLOOR, led.ROUND_BUDGET,
            led.ONE_SHOT_REFUSED, led.FIXED_LADDER_EXHAUSTED, led.LOSSLESS_ENTRY,
        }

    def test_a_step_whose_tuner_never_ran_seals_an_empty_ledger(self, mock_pipeline):
        from mimarsinan.pipelining.core.steps.tuner_pipeline_step import (
            TunerPipelineStep,
        )

        class _NoTunerStep(TunerPipelineStep):
            def __init__(self, pipeline):
                super().__init__((), self.PROMISES, (), (), pipeline)

            def process(self):
                pass

        step = _NoTunerStep(mock_pipeline)
        step.name = "No Tuner"
        mock_pipeline.prepare_step(step)
        step.run()
        payload = mock_pipeline.cache[f"No Tuner.{led.LEDGER_ENTRY_KEY}"]
        assert payload == AdaptationLedger().to_dict()


# --------------------------------------------------------------------------- #
# Event shapes: every group is a frozen dataclass with a JSON-safe to_dict.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "event",
    [
        Proposal(cycle_index=0, rate=0.5, committed=0.0, increment=0.5),
        Verdict(
            cycle_index=0, outcome=led.COMMIT, accepted=True, rolled_back=False,
            probe_metric=PROBE_MARGINAL, reading=0.9, threshold=0.8, probe_reads=2,
        ),
        Refinement(kind=led.BISECT_STEP, detail="0.5 -> 0.25"),
        Escalation(path=led.EPSILON_FLOOR, detail="underflow"),
        Recovery(cycle_index=0, kind=led.RECOVERY_TRAINED, steps=10, lr=1e-3),
        Endpoint(
            stage="T", steps=1, budget_steps=2, engaged=True, armed=False,
            reached=True,
        ),
    ],
    ids=lambda e: type(e).__name__,
)
def test_event_to_dict_is_json_safe(event):
    payload = event.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {f.name for f in dataclasses.fields(event)}
