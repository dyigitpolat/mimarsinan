"""[MBH-ENDPOINT] P1'': bounded train-to-target on the deployed composition at conversion endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration import dhat_highwater
from mimarsinan.tuning.orchestration.mbh_ledger import fp32_eval_forward_over_val
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


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


def freed_ladder_steps(tuner) -> int:
    """Training steps the tuner's own fixed ladder left unconsumed (stalls/rejects).

    Gate refinements can train MORE than the planned ladder; the freed budget is
    never negative.
    """
    rates = getattr(tuner, "_fixed_ladder_rates", None) or []
    planned = len(rates) * int(getattr(tuner, "_fast_steps_per_rate", 0))
    used = int(getattr(tuner, "_fast_optimizer_steps", 0))
    return max(0, planned - used)


def _fp32_deployed_read(tuner) -> float:
    """fp32 accuracy of the LIVE model (the deployed composition at an endpoint)
    over the tuner's eval batches — like-for-like with the gate's D-hat reads."""
    device = tuner.pipeline.config["device"]
    return float(fp32_eval_forward_over_val(
        tuner.trainer, tuner.model, tuner.model,
        tuner._budget.eval_n_batches, device,
    ))


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
      whatever this tuner's own gated ladder left untrained.
    - geometry: a floor-LIFTED target trains the probe-validated arm (lr capped
      at ``endpoint_floor_lr``, cosine over the full budget with patience
      disarmed — the default 210-step window is sterile inside the lr dip);
      otherwise the stage is bit-identical to the pre-floor behavior.
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
    steps_used = 0
    exit_read = entry
    rolled_back = False
    if engaged:
        pre_state = tuner._clone_state()
        hooks = tuner._recovery_training_hooks(1.0)
        lr = float(tuner.pipeline_lr)
        min_steps = 0
        max_seconds = None
        if floor_lifted:
            lr = min(lr, TUNING_POLICY.endpoint_floor_lr)
            min_steps = budget
            max_seconds = float(tuner.pipeline.config.get(
                "endpoint_floor_wall_s", TUNING_POLICY.endpoint_floor_wall_s,
            ))
        _, steps_used = RecoveryEngine.train_to_target(
            tuner.trainer,
            lr,
            target,
            max_steps=budget,
            hooks=hooks,
            validation_n_batches=tuner._budget.progress_eval_batches,
            check_interval=tuner._budget.check_interval,
            patience=_RECOVERY_PATIENCE,
            min_steps=min_steps,
            min_improvement=tuner._budget.accuracy_se(),
            cosine_decay=True,
            return_steps=True,
            final_validation=False,
            max_seconds=max_seconds,
        )
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
    )
    _emit(tuner, report)
    return report


def _emit(tuner, report: EndpointRecoveryReport) -> None:
    print(
        f"[MBH-ENDPOINT] tuner={type(tuner).__name__} "
        f"target={report.target:.6f} entry={report.entry:.6f} "
        f"exit={report.exit:.6f} budget={report.budget_steps} "
        f"steps_used={report.steps_used} engaged={report.engaged} "
        f"reached={report.reached} rolled_back={report.rolled_back} "
        f"target_floor={report.target_floor:.6f} "
        f"floor_lifted={report.floor_lifted}",
        flush=True,
    )
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
    })
