"""[MBH-GATE] D-hat-gated fast ladder — opt-in trust-region ratchet (X2/E1, T3)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

from mimarsinan.tuning.orchestration import mbh_ledger

ACCEPT_TOLERANCE = 0.01
MAX_REFINEMENTS = 3


@dataclass
class MBHGateState:
    """Per-run gate scratch: the D-hat ratchet anchor and its model snapshot."""

    best_full_acc: float
    best_state: Any
    stalled: bool = False
    rung: int = -1


def gated_fast_rate_attempt(tuner, target: float) -> float:
    """One D-hat-gated fast-ladder rung.

    Snapshot -> train the rung -> measure the deployed full-transform accuracy
    (fp32, clone-based) -> ACCEPT iff D-hat >= best - ACCEPT_TOLERANCE, else
    restore and retry the midpoint rate (max MAX_REFINEMENTS). Exhaustion is a
    CONSTRUCTIVE STALL: stop consuming rungs, restore the best-D-hat snapshot.
    """
    state = _ensure_gate_state(tuner)
    if state.stalled:
        return float(tuner._committed_rate)
    tuner._ensure_fast_optimizer()
    state.rung += 1
    committed_before = float(tuner._committed_rate)
    rate = float(target)
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
        if full_acc >= state.best_full_acc - ACCEPT_TOLERANCE:
            _accept(tuner, state, rate, post_acc, full_acc, t0)
            return rate
        _restore_live(tuner, snapshot)
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
            f"full_acc={full_acc:.6f} best_full_acc={state.best_full_acc:.6f} "
            f"{tail}",
        )
        rate = retry
    state.stalled = True
    print(
        f"[MBH-GATE] constructive_stall committed={committed_before:.6f} "
        f"best_full_acc={state.best_full_acc:.6f}",
        flush=True,
    )
    tuner._restore_state(state.best_state)
    return committed_before


def _ensure_gate_state(tuner) -> MBHGateState:
    """Lazily anchor the ratchet on the ENTRY D-hat (the debt is largest at the
    first rung, where the blended gate is most confidently wrong — X1 §5b)."""
    state = getattr(tuner, "_mbh_gate_state", None)
    if state is None:
        entry = float(mbh_ledger.full_transform_measurement(tuner))
        state = MBHGateState(best_full_acc=entry, best_state=tuner._clone_state())
        tuner._mbh_gate_state = state
        _log(tuner, f"entry best_full_acc={entry:.6f}")
    return state


def _accept(tuner, state, rate, post_acc, full_acc, t0) -> None:
    """Commit the rung exactly like the ungated attempt, then ratchet best-D-hat."""
    tuner._committed_rate = float(rate)
    tuner._record_fast_cycle(rate, post_acc, t0)
    tuner._last_post_acc = post_acc
    tuner._fast_probe(float(rate))
    _add_phase_seconds(tuner, t0)
    if full_acc >= state.best_full_acc:
        state.best_full_acc = float(full_acc)
        state.best_state = tuner._clone_state()
    _log(
        tuner,
        f"accept rung={state.rung} rate={float(rate):.6f} "
        f"full_acc={full_acc:.6f} best_full_acc={state.best_full_acc:.6f}",
    )


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
