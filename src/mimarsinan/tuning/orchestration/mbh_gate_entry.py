"""[MBH-GATE] the gate's ENTRY: what the ratchet anchors on, and whether the ladder is needed at all."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.tuning.orchestration import dhat_highwater, mbh_ledger
from mimarsinan.tuning.orchestration.retention_envelope import resolve_step_anchor
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


def entry_is_lossless(entry_full: float, pre_transform: float, se: float) -> bool:
    """[recipe-economics] the transform-damage predicate: the FULL transform's
    entry read within ``lossless_entry_se_margin`` SE of the PRE-TRANSFORM read
    means the ladder/ramp has nothing to smooth.

    The reference is pre-transform by necessity. A realizable ramp (the LIF
    T-anneal) pins its blend at 1.0 and puts the model in the TRANSFORMED state
    at rate 0, so the tuner's own entry probe measures the damage against
    itself: it called a measured 0.9823 -> 0.7730 conversion lossless and
    skipped the whole LIF ladder.
    """
    margin = float(TUNING_POLICY.lossless_entry_se_margin) * float(se)
    return float(entry_full) >= float(pre_transform) - margin


def pre_transform_reference(tuner) -> float | None:
    """The read the transform is charged against: the metric the model carried
    INTO this step (``resolve_step_anchor``, the same anchor the step's floors
    and targets use). ``None`` when the pipeline carries no anchor."""
    anchor = resolve_step_anchor(tuner.pipeline)
    if anchor is None or float(anchor) <= 0.0:
        return None
    return float(anchor)


def lossless_entry_short_circuit(tuner) -> bool:
    """Driver predicate: measure the entry (isolation-guarded, memoized in the
    gate state), decide, and on a lossless entry commit rate 1.0 for finalize.
    Loud by contract — a skipped ladder must never be silent."""
    state = _ensure_gate_state(tuner)
    if state.pre_transform_acc is None:
        _log(
            tuner,
            "entry_fast_path: refused — the pipeline carries no pre-transform "
            "anchor, so this transform's cost cannot be measured; running the "
            "ladder",
        )
        return False
    pre = float(state.pre_transform_acc)
    se = float(tuner._budget.accuracy_se())
    if not entry_is_lossless(state.best_full_acc, pre, se):
        return False
    tuner._committed_rate = 1.0
    tuner._entry_short_circuited = True
    _log(
        tuner,
        f"entry_fast_path: full={state.best_full_acc:.6f} "
        f"pre_transform={pre:.6f} se={se:.6f} — transform lossless at "
        f"entry; ladder+ramp skipped, finalize at rate 1.0",
    )
    _event(
        tuner, "entry_fast_path", full_acc=float(state.best_full_acc),
        pre_transform_acc=pre, post_acc=float(state.prev_post_acc or 0.0), se=se,
    )
    return True
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
    # The read the model carried INTO this step — the reference the entry
    # fast path charges the transform against (never the tuner's own entry
    # probe, which a realizable ramp has already transformed).
    pre_transform_acc: float | None = None
    # Best DEPLOYED (full-transform) read over ALL attempts, accepted or
    # rejected — the finalize-arbitration candidate (retention can rightly
    # reject the deploy-best state mid-ladder; see finalize_on_best_deployed).
    best_deployed_acc: float = float("-inf")
    best_deployed_state: Any = None


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
        pre_transform = pre_transform_reference(tuner)
        state = MBHGateState(
            best_full_acc=entry, best_state=entry_state,
            prev_post_acc=entry_post, retention_armed=armed,
            best_deployed_acc=entry, best_deployed_state=entry_state,
            pre_transform_acc=pre_transform,
        )
        tuner._mbh_gate_state = state
        _log(
            tuner,
            f"entry best_full_acc={entry:.6f} post_acc={entry_post:.6f} "
            f"pre_transform_acc={pre_transform} retention_armed={armed}",
        )
        _event(
            tuner, "entry", best_full_acc=entry, post_acc=entry_post,
            pre_transform_acc=pre_transform, retention_armed=armed,
        )
    return state


def _log(tuner, message: str) -> None:
    print(f"[MBH-GATE] tuner={type(tuner).__name__} {message}", flush=True)


def _event(tuner, action: str, **payload) -> None:
    emit_reporter_event(
        tuner.pipeline.reporter,
        "mbh_gate", {"action": action, "tuner": type(tuner).__name__, **payload},
    )
