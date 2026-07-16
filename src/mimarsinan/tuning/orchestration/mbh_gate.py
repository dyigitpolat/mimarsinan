"""[MBH-GATE] D-hat-gated fast ladder — the default trust-region driver (X3, from X2/T3)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.tuning.orchestration import dhat_highwater, mbh_ledger

ACCEPT_TOLERANCE = 0.01
MAX_REFINEMENTS = 3
def _retention_tolerance(tuner) -> float:
    """Never demand retention finer than the metric's own noise: 2x the
    Bernoulli SE of the gate's eval window, floored at ACCEPT_TOLERANCE."""
    return max(ACCEPT_TOLERANCE, 2.0 * float(tuner._budget.accuracy_se()))


def _retention_armed(tuner, entry_post: float) -> bool:
    """Retention arms only above the pretrain chance envelope (the engine's
    ``pretrain_floor_chance_multiple`` SSOT): a relative retention bound on a
    chance-level backbone gates pure noise."""
    num_classes = tuner.pipeline.config.get("num_classes")
    if not num_classes or int(num_classes) <= 1:
        return True
    multiple = float(
        tuner.pipeline.config.get("pretrain_floor_chance_multiple", 5.0)
    )
    return float(entry_post) >= multiple / float(num_classes)


@dataclass
class MBHGateState:
    """Per-run gate scratch: the D-hat ratchet anchor and its model snapshot."""

    best_full_acc: float
    best_state: Any
    stalled: bool = False
    rung: int = -1
    prev_post_acc: float | None = None
    retention_armed: bool = False
    # Best DEPLOYED (full-transform) read over ALL attempts, accepted or
    # rejected — the finalize-arbitration candidate (retention can rightly
    # reject the deploy-best state mid-ladder; see finalize_on_best_deployed).
    best_deployed_acc: float = float("-inf")
    best_deployed_state: Any = None


def gated_fast_rate_attempt(tuner, target: float) -> float:
    """One two-sided-gated fast-ladder rung.

    Snapshot -> train the rung -> measure the deployed full-transform accuracy
    (fp32, clone-based) -> ACCEPT iff D-hat >= best - ACCEPT_TOLERANCE AND the
    blended post_acc retained the previous commit (>= prev - ACCEPT_TOLERANCE;
    training at an oversized LR wrecked a ViT blend 0.82->0.31 while D-hat
    improved — the one-sided gate accepted the wreck). A retention reject also
    halves the ladder LR (Armijo backoff: destruction is an LR problem; a
    D-hat reject is a RATE problem and keeps the LR); both retry the midpoint
    rate (max MAX_REFINEMENTS). The drifting anchor admits at most
    ACCEPT_TOLERANCE per rung — worst-case n_rungs*tol, vs the measured 51-pt
    destruction. Exhaustion is a CONSTRUCTIVE STALL: stop consuming rungs,
    restore the best-D-hat snapshot.
    """
    state = _ensure_gate_state(tuner)
    if state.stalled:
        return float(tuner._committed_rate)
    tuner._ensure_fast_optimizer()
    state.rung += 1
    committed_before = float(tuner._committed_rate)
    rate = float(target)
    tuner._fast_retry_step_scale = 1
    for attempt in range(1 + MAX_REFINEMENTS):
        snapshot = _snapshot_live(tuner)
        t0 = time.time()
        tuner._fast_ramp(rate)
        post_acc = float(tuner.probe())
        measurements = mbh_ledger.rung_measurements(tuner)
        mbh_ledger.emit_fast_rung_ledger(
            tuner, rate=rate, blended_acc=post_acc, measurements=measurements,
        )
        full_acc = float(measurements["full_acc"])
        dhat_highwater.observe(tuner.pipeline, full_acc)
        if full_acc > state.best_deployed_acc:
            state.best_deployed_acc = float(full_acc)
            state.best_deployed_state = tuner._clone_state()
        dhat_ok = full_acc >= state.best_full_acc - ACCEPT_TOLERANCE
        retained = (
            not state.retention_armed
            or state.prev_post_acc is None
            or post_acc >= state.prev_post_acc - _retention_tolerance(tuner)
        )
        if dhat_ok and retained:
            _accept(tuner, state, rate, post_acc, full_acc, t0)
            return rate
        reason = "dhat" if not dhat_ok else "retention"
        _restore_live(tuner, snapshot)
        if reason == "retention":
            # Armijo trust region preserving LR x steps: halve the step SIZE,
            # double the step COUNT (fixed-step halvings converged
            # 0.09->0.39->0.67->0.77 and exhausted short of the bound).
            tuner._scale_fast_lr(0.5)
            tuner._fast_retry_step_scale = min(
                8, 2 * max(1, int(getattr(tuner, "_fast_retry_step_scale", 1)))
            )
        tuner._record_fast_cycle(rate, post_acc, t0, outcome="rollback")
        _add_phase_seconds(tuner, t0)
        retry = (committed_before + rate) / 2.0
        tail = (
            f"retry_rate={retry:.6f}" if attempt < MAX_REFINEMENTS
            else "refinements_exhausted"
        )
        _log(
            tuner,
            f"reject rung={state.rung} attempt={attempt} rate={rate:.6f} "
            f"reason={reason} post_acc={post_acc:.6f} "
            f"full_acc={full_acc:.6f} best_full_acc={state.best_full_acc:.6f} "
            f"{tail}",
        )
        _event(
            tuner, "reject", rung=state.rung, attempt=attempt, rate=float(rate),
            reason=reason, post_acc=float(post_acc),
            full_acc=float(full_acc), best_full_acc=float(state.best_full_acc),
        )
        rate = retry
    state.stalled = True
    print(
        f"[MBH-GATE] constructive_stall committed={committed_before:.6f} "
        f"best_full_acc={state.best_full_acc:.6f}",
        flush=True,
    )
    _event(
        tuner, "stall", rung=state.rung, committed=float(committed_before),
        best_full_acc=float(state.best_full_acc),
    )
    tuner._restore_state(state.best_state)
    return committed_before


def _ensure_gate_state(tuner) -> MBHGateState:
    """Lazily anchor the ratchet on the ENTRY D-hat (the debt is largest at the
    first rung, where the blended gate is most confidently wrong — X1 §5b) and
    the retention anchor on the ENTRY blended read (isolation-guarded: the
    extra probe must not perturb the live RNG/cursor trajectory)."""
    state = getattr(tuner, "_mbh_gate_state", None)
    if state is None:
        entry = float(mbh_ledger.full_transform_measurement(tuner))
        dhat_highwater.observe(tuner.pipeline, entry)
        with mbh_ledger._measurement_guard(tuner.trainer):
            entry_post = float(tuner.probe())
        armed = _retention_armed(tuner, entry_post)
        entry_state = tuner._clone_state()
        state = MBHGateState(
            best_full_acc=entry, best_state=entry_state,
            prev_post_acc=entry_post, retention_armed=armed,
            best_deployed_acc=entry, best_deployed_state=entry_state,
        )
        tuner._mbh_gate_state = state
        _log(
            tuner,
            f"entry best_full_acc={entry:.6f} post_acc={entry_post:.6f} "
            f"retention_armed={armed}",
        )
        _event(
            tuner, "entry", best_full_acc=entry, post_acc=entry_post,
            retention_armed=armed,
        )
    return state


def _accept(tuner, state, rate, post_acc, full_acc, t0) -> None:
    """Commit the rung exactly like the ungated attempt, then ratchet best-D-hat."""
    tuner._committed_rate = float(rate)
    tuner._record_fast_cycle(rate, post_acc, t0)
    tuner._last_post_acc = post_acc
    tuner._fast_probe(float(rate))
    _add_phase_seconds(tuner, t0)
    tuner._fast_retry_step_scale = 1
    state.prev_post_acc = float(post_acc)
    if full_acc >= state.best_full_acc:
        state.best_full_acc = float(full_acc)
        state.best_state = tuner._clone_state()
    _log(
        tuner,
        f"accept rung={state.rung} rate={float(rate):.6f} "
        f"full_acc={full_acc:.6f} best_full_acc={state.best_full_acc:.6f}",
    )
    _event(
        tuner, "accept", rung=state.rung, rate=float(rate),
        full_acc=float(full_acc), best_full_acc=float(state.best_full_acc),
    )


def finalize_on_best_deployed(tuner):
    """[WS-A A1] fixed-ladder finalize arbitration: a ladder that ends below
    its target rate hard-finalizes from a state whose deployed read can sit
    far below a mid-ladder candidate (measured: a rate-1.0 attempt read 0.7515
    full-ReLU and was retention-rejected; the sub-1.0 commit's hard swap read
    0.06). Restores the best deployed state when the FINAL state's deployed
    read falls short of it by more than ACCEPT_TOLERANCE; returns the restored
    read, or None when the final state stands (healthy ladders are inert:
    their last accept IS the deployed best)."""
    state = getattr(tuner, "_mbh_gate_state", None)
    if state is None or state.best_deployed_state is None:
        return None
    final_read = float(mbh_ledger.full_transform_measurement(tuner))
    if final_read >= state.best_deployed_acc - ACCEPT_TOLERANCE:
        return None
    tuner._restore_state(state.best_deployed_state)
    _log(
        tuner,
        f"finalize_best_deployed restored={state.best_deployed_acc:.6f} "
        f"final_read={final_read:.6f}",
    )
    _event(
        tuner, "finalize_best_deployed",
        restored=float(state.best_deployed_acc), final_read=float(final_read),
    )
    return float(state.best_deployed_acc)


def _snapshot_live(tuner) -> tuple:
    """Model+axis state (CheckpointGuard) plus the shared fast optimizer/schedule."""
    return (
        tuner._clone_state(),
        copy.deepcopy(tuner._fast_optimizer.state_dict()),
        copy.deepcopy(tuner._fast_lr_schedule.state_dict()),
        int(tuner._fast_optimizer_steps),
    )


def _restore_live(tuner, snapshot) -> None:
    live_state, optimizer_sd, schedule_sd, optimizer_steps = snapshot
    tuner._restore_state(live_state)
    tuner._fast_optimizer.load_state_dict(optimizer_sd)
    tuner._fast_lr_schedule.load_state_dict(schedule_sd)
    tuner._fast_optimizer_steps = optimizer_steps


def _add_phase_seconds(tuner, t0) -> None:
    tuner._phase_seconds["fast_blend"] = (
        tuner._phase_seconds.get("fast_blend", 0.0) + (time.time() - t0)
    )


def _log(tuner, message: str) -> None:
    print(f"[MBH-GATE] tuner={type(tuner).__name__} {message}", flush=True)


def _event(tuner, action: str, **payload) -> None:
    emit_reporter_event(
        tuner.pipeline.reporter,
        "mbh_gate", {"action": action, "tuner": type(tuner).__name__, **payload},
    )
