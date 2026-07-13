"""[MBH-ENDPOINT] P1'': bounded train-to-target on the deployed composition at conversion endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_steps
from mimarsinan.tuning.orchestration.frontier.divergence_guard import (
    DivergenceGuard,
    rescue_plan,
)
from mimarsinan.tuning.orchestration.mbh_ledger import fp32_deployed_read
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE
from mimarsinan.tuning.orchestration.tuning_policy import (
    TUNING_POLICY,
    armed_endpoint_effective_check_interval,
    effective_endpoint_floor_lr,
    endpoint_convergence_geometry,
)


@dataclass(frozen=True)
class EndpointRecoveryReport:
    """One endpoint-stage engagement record (target, reads, budget, verdict)."""

    target: float
    entry: float
    exit: float
    budget_steps: int
    steps_used: int
    engaged: bool
    reached: bool
    rolled_back: bool
    target_floor: float = 0.0
    floor_lifted: bool = False
    armed: bool = False
    divergence_rescued: bool = False


def freed_ladder_steps(tuner) -> int:
    """Training steps the tuner's own fixed ladder left unconsumed (stalls/rejects).

    Gate refinements can train MORE than the planned ladder; the freed budget is
    never negative.
    """
    rates = getattr(tuner, "_fixed_ladder_rates", None) or []
    planned = len(rates) * int(getattr(tuner, "_fast_steps_per_rate", 0))
    used = int(getattr(tuner, "_fast_optimizer_steps", 0))
    return max(0, planned - used)


# Module-level alias: tests monkeypatch this seam; the shared implementation
# lives in mbh_ledger (also the [MBH-DRAWS] selection read).
_fp32_deployed_read = fp32_deployed_read


def run_endpoint_recovery(tuner, *, base_steps, target_floor=None) -> EndpointRecoveryReport:
    """The generic P1'' endpoint stage: after a conversion tuner reaches rate 1.0
    and finalizes, train the DEPLOYED-COMPOSITION forward to the pipeline D-hat
    high-water target, bounded and rollback-guarded (never ends below entry).

    - target: max(D-hat high-water SSOT, the [5u] target floor). ``target_floor``
      lets a call site pass an explicit floor (the WQ endpoint scopes the [5u
      generalized] well-conditioned floor to the FINAL composition only); when
      ``None`` the floor reads ``endpoint_target_floor`` from config — the
      bit-parity-lossless family's every-endpoint floor.
    - budget: ``base_steps`` (the recipe's freed stabilize/ladder budget) plus
      whatever this tuner's own gated ladder left untrained; an armed stage is
      additionally clamped to what the run's ``endpoint_floor_steps`` ledger
      still affords.
    - geometry [C2 arm-when-engaged]: EVERY engaged stage the ledger affords
      trains the probe-validated funded arm — lr capped at
      ``endpoint_floor_lr``, cosine, step-ledger-funded, with the [C1]
      convergence stop (min-cover = max(absolute lr-dip cover, fraction of
      budget), keep-best patience scaled to the budget) so the budget is a
      true CEILING, never a mandatory burn; [P4] the funded leg evals at the
      budget-aware cadence (``armed_endpoint_effective_check_interval``:
      widened at/above the [C1] min-cover, DENSE below it — coarse keep-best
      sampling missed the trajectory peak on the v6 small-budget cells) with
      the geometry composed at that same interval, while a non-armed stage
      keeps the exact
      ``_RECOVERY_PATIENCE x check_interval`` stagnation economics
      (mbh_analytical_ttfs_stagnation). The former >=1xSE entry-gap gate
      is deleted: it left sub-SE engaged stages sterile (t0_14's 210-step
      +0.0000 window sits inside the lr dip) and armed razor-edge cells
      nondeterministically (t01_06 margin 0.000186). Only a stage the
      exhausted ledger no longer funds keeps the pre-floor patience geometry
      (cheap stop).
    - [C3] with ``TUNING_POLICY.endpoint_floor_divergence_rescue`` on, an
      armed leg is watched by the dead-run predicate (never beat entry+SE /
      pipeline-hard-floor crater); a fired leg restores its live keep-best,
      rebuilds the optimizer, and restarts the remaining budget once at
      lr*0.3 with a warmup ramp. Flag-off is byte-identical.
    - reproducibility: the funding is a STEP budget, never wall seconds —
      identical configs train identical step counts on any hardware (same
      config + same seed => same step trajectory, modulo GPU nondeterminism);
      wall time is a pure measurement.
    - keep-best + early stop live inside ``train_steps_until_target``; the
      outer fp32 entry guard makes the whole stage non-destructive.
    """
    highwater = dhat_highwater.require(tuner.pipeline)
    if target_floor is None:
        floor = float(tuner.pipeline.config.get("endpoint_target_floor", 0.0))
    else:
        floor = float(target_floor)
    target = max(highwater, floor)
    floor_lifted = target > highwater
    budget = int(base_steps) + freed_ladder_steps(tuner)
    entry = _fp32_deployed_read(tuner)

    engaged = budget > 0 and entry < target
    armed = False
    rescued = False
    steps_used = 0
    exit_read = entry
    rolled_back = False
    trajectory: list[tuple[int, float, float]] = []
    if engaged:
        pre_state = tuner._clone_state()
        lr = float(tuner.pipeline_lr)
        min_steps = 0
        patience = _RECOVERY_PATIENCE
        check_interval = int(tuner._budget.check_interval)
        # ``endpoint_floor_steps`` is the RUN total, shared by every armed
        # endpoint stage through the step ledger (per-stage budgets burned
        # budget x N stages on multi-endpoint cells); an exhausted budget
        # falls back to the default patience geometry (cheap stop).
        total_steps = int(tuner.pipeline.config.get(
            "endpoint_floor_steps", TUNING_POLICY.endpoint_floor_steps,
        ))
        steps_left = endpoint_steps.remaining(tuner.pipeline, total_steps)
        if steps_left > 0:
            lr = min(lr, effective_endpoint_floor_lr(tuner.pipeline.config))
            budget = min(budget, steps_left)
            # [P4] the funded leg evals at the budget-aware effective cadence
            # (dense below the [C1] min-cover, widened at/above); the geometry
            # composes at the SAME interval (patience counts checks), so the
            # patience step-window stays ~fraction x budget, never multiplied.
            check_interval = armed_endpoint_effective_check_interval(
                budget, check_interval,
            )
            geometry = endpoint_convergence_geometry(budget, check_interval)
            min_steps = geometry.min_steps
            patience = geometry.patience
            armed = True
        steps_used, rescued = _train_engaged(
            tuner, lr=lr, target=target, budget=budget, min_steps=min_steps,
            patience=patience, armed=armed, check_interval=check_interval,
            trajectory=trajectory,
        )
        if armed:
            endpoint_steps.consume(tuner.pipeline, steps_used)
        exit_read = _fp32_deployed_read(tuner)
        tol = float(getattr(tuner, "_rollback_tolerance", 0.0))
        if exit_read < entry - tol:
            tuner._restore_state(pre_state)
            exit_read = entry
            rolled_back = True

    dhat_highwater.observe(tuner.pipeline, exit_read)
    report = EndpointRecoveryReport(
        target=float(target),
        entry=float(entry),
        exit=float(exit_read),
        budget_steps=int(budget),
        steps_used=int(steps_used),
        engaged=bool(engaged),
        reached=bool(exit_read >= target),
        rolled_back=bool(rolled_back),
        target_floor=float(floor),
        floor_lifted=bool(floor_lifted),
        armed=bool(armed),
        divergence_rescued=bool(rescued),
    )
    _emit(tuner, report, trajectory)
    return report


def _train_engaged(tuner, *, lr, target, budget, min_steps, patience, armed,
                   check_interval, trajectory):
    """One engaged endpoint training leg (+ the [C3] guarded rescue leg when
    armed and the rescue flag is on); returns ``(steps_used, rescued)``."""
    def record(step, acc, best_acc, entry_acc):
        trajectory.append((int(step), float(acc), float(best_acc)))
        return False

    guard = None
    if armed and TUNING_POLICY.endpoint_floor_divergence_rescue:
        guard = DivergenceGuard(
            accuracy_se=float(tuner._budget.accuracy_se()),
            hard_floor=getattr(tuner, "_pipeline_hard_floor", None),
        )

    def on_check(step, acc, best_acc, entry_acc):
        record(step, acc, best_acc, entry_acc)
        if guard is None:
            return False
        return guard(step, acc, best_acc, entry_acc)

    _, steps_used = RecoveryEngine.train_to_target(
        tuner.trainer,
        lr,
        target,
        max_steps=budget,
        hooks=tuner._recovery_training_hooks(1.0),
        validation_n_batches=tuner._budget.progress_eval_batches,
        check_interval=check_interval,
        patience=patience,
        min_steps=min_steps,
        min_improvement=tuner._budget.accuracy_se(),
        cosine_decay=True,
        return_steps=True,
        final_validation=False,
        on_check=on_check,
    )
    steps_used = int(steps_used)
    if guard is None or not guard.fired:
        return steps_used, False
    plan = rescue_plan(budget - steps_used, lr)
    if plan is None:
        return steps_used, False
    # [C3] the fired leg already restored its live keep-best (the loop's
    # entry-anchored restore); the restart builds a fresh optimizer. The
    # rescue leg is armed-only, so it keeps the stage's [P4] effective cadence.
    geometry = endpoint_convergence_geometry(plan.train_steps, check_interval)
    _, rescue_steps = RecoveryEngine.train_to_target(
        tuner.trainer,
        plan.lr,
        target,
        max_steps=plan.train_steps,
        warmup_steps=plan.warmup_steps,
        hooks=tuner._recovery_training_hooks(1.0),
        validation_n_batches=tuner._budget.progress_eval_batches,
        check_interval=check_interval,
        patience=geometry.patience,
        min_steps=geometry.min_steps,
        min_improvement=tuner._budget.accuracy_se(),
        cosine_decay=True,
        return_steps=True,
        final_validation=False,
        on_check=record,
    )
    return steps_used + int(rescue_steps), True


def _emit(tuner, report: EndpointRecoveryReport, trajectory) -> None:
    print(
        f"[MBH-ENDPOINT] tuner={type(tuner).__name__} "
        f"target={report.target:.6f} entry={report.entry:.6f} "
        f"exit={report.exit:.6f} budget={report.budget_steps} "
        f"steps_used={report.steps_used} engaged={report.engaged} "
        f"reached={report.reached} rolled_back={report.rolled_back} "
        f"target_floor={report.target_floor:.6f} "
        f"floor_lifted={report.floor_lifted} "
        f"armed={report.armed} "
        f"divergence_rescued={report.divergence_rescued}",
        flush=True,
    )
    emit_reporter_event(tuner.pipeline.reporter, "mbh_endpoint", {
        "tuner": type(tuner).__name__,
        "target": report.target,
        "entry": report.entry,
        "exit": report.exit,
        "budget_steps": report.budget_steps,
        "steps_used": report.steps_used,
        "engaged": report.engaged,
        "reached": report.reached,
        "rolled_back": report.rolled_back,
        "target_floor": report.target_floor,
        "floor_lifted": report.floor_lifted,
        "armed": report.armed,
        "divergence_rescued": report.divergence_rescued,
        "trajectory": list(trajectory),
    })
    tuner.pipeline.reporter.report(f"{tuner.name} endpoint_recovery", {
        "target": round(report.target, 4),
        "entry": round(report.entry, 4),
        "exit": round(report.exit, 4),
        "budget_steps": report.budget_steps,
        "steps_used": report.steps_used,
        "engaged": report.engaged,
        "reached": report.reached,
        "rolled_back": report.rolled_back,
        "target_floor": round(report.target_floor, 4),
        "floor_lifted": report.floor_lifted,
        "armed": report.armed,
        "divergence_rescued": report.divergence_rescued,
    })
