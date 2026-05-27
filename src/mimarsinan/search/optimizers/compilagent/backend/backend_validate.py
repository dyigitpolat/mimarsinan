"""Intervention validation for MimarsinanLayoutBackend."""

from __future__ import annotations

from compilagent import Intervention, ValidationResult

from ..plan_codec import ARCH_KIND, HW_CORE_KIND, HW_DIM_NAMES


def validate_intervention(intervention: Intervention) -> ValidationResult:
    target = intervention.target
    if target.kind not in (ARCH_KIND, HW_CORE_KIND):
        return ValidationResult(
            ok=False,
            errors=(
                f"unknown target.kind {target.kind!r}; expected "
                f"{ARCH_KIND!r} or {HW_CORE_KIND!r}",
            ),
        )
    if target.kind == ARCH_KIND:
        if not target.selector:
            return ValidationResult(
                ok=False,
                errors=("arch intervention requires a non-empty selector",),
            )
        return ValidationResult(ok=True)
    parts = target.selector.split(".")
    if len(parts) != 2 or parts[1] not in HW_DIM_NAMES:
        dim_options = "|".join(HW_DIM_NAMES)
        return ValidationResult(
            ok=False,
            errors=(
                f"hw.core selector must be `<core_index>.{{{dim_options}}}`, "
                f"got {target.selector!r}",
            ),
        )
    try:
        int(parts[0])
        value = int(intervention.payload)
    except (TypeError, ValueError):
        return ValidationResult(
            ok=False,
            errors=(
                f"hw.core core index and payload must be integers, "
                f"got {parts[0]!r}/{intervention.payload!r}",
            ),
        )
    if value <= 0:
        return ValidationResult(
            ok=False,
            errors=(
                f"hw.core payload for {target.selector} must be a "
                f"positive integer, got {value}",
            ),
        )
    hard_max = 65536
    if value > hard_max:
        return ValidationResult(
            ok=False,
            errors=(
                f"hw.core payload for {target.selector} must be "
                f"<= {hard_max}, got {value}",
            ),
        )
    if parts[1] in ("max_axons", "max_neurons") and value % 8 != 0:
        snap_lo = (value // 8) * 8
        snap_hi = ((value + 7) // 8) * 8
        return ValidationResult(
            ok=False,
            errors=(
                f"hw.core payload for {target.selector} should be a "
                f"multiple of 8 (the layout packer pads otherwise); "
                f"got {value}. Snap to {snap_lo} or {snap_hi}.",
            ),
        )
    return ValidationResult(ok=True)
