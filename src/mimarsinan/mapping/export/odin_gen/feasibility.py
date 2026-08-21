"""The generated core's feasibility gates: theta width and the no-saturation bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from mimarsinan.mapping.export.odin.feasibility import OdinFeasibilityError
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec

KEY_VARIANT_THETA = "odin_gen.theta_range"
KEY_NO_SATURATION = "odin_gen.no_saturation"


@dataclass(frozen=True)
class SaturationBound:
    """One core's provable membrane interval over a whole sample."""

    core_index: int
    lowest: int
    highest: int

    def inside(self, spec: CoreSpec) -> bool:
        """STRICTLY inside the register: a rail reached is a rail touched."""
        return spec.membrane_low < self.lowest and self.highest < spec.membrane_high


def check_variant_theta(theta: Any, *, spec: CoreSpec, core_index: int) -> int:
    """Theta shares the membrane register's width, so it shares its interval."""
    value = _as_integer(theta)
    if value is None:
        raise OdinFeasibilityError(
            KEY_VARIANT_THETA,
            f"core {core_index}: threshold {theta!r} is not integral; the "
            f"threshold register is a {spec.membrane_bits}-bit integer and a "
            f"silent truncation would change the deployed physics.")
    if value < 1 or value > spec.membrane_high:
        raise OdinFeasibilityError(
            KEY_VARIANT_THETA,
            f"core {core_index}: threshold {value} is outside "
            f"[1, {spec.membrane_high}], the interval its "
            f"{spec.membrane_bits}-bit register holds. theta IS the folded "
            f"quantization scale, so this is a real scope limit: re-run the "
            f"scale/threshold adaptation ladder, or generate a wider membrane.")
    return value


def core_saturation_bound(
    core: Any, *, spec: CoreSpec, core_index: int, theta: int, cycles: int,
    membrane_init: int,
) -> SaturationBound:
    """A STATIC bound on the membrane of every neuron of one per-cycle core.

    Per-cycle law, so each neuron fires at most once per cycle and every wire
    into the core therefore carries at most one event per cycle. Writing
    ``pos``/``neg`` for a neuron's positive/negative weight sums:

      * the in-cycle PEAK is ``(theta - 1) + pos``: the membrane entering a
        cycle is always below theta (every fire-path leaves it there, and the
        window starts at ``0 <= V0 < theta``), and the whole positive drive
        lands before the single compare;
      * the FLOOR is ``min(V0, V0 + cycles * neg)``: nothing resets a membrane
        upward except a fire, which raises it, so the deepest reachable value
        is the one that never fires and takes the full negative drive every
        cycle.

    Both are bounds on the UNCLAMPED value the register would have to hold, so
    proving them strictly inside the register's interval proves this law holds
    exactly the number the unbounded accumulator holds.
    """
    matrix = np.rint(
        np.asarray(core.get_core_matrix()).astype(np.float64)).astype(np.int64)
    used_neurons = int(core.neurons_per_core) - int(core.available_neurons or 0)
    if used_neurons <= 0 or matrix.size == 0:
        return SaturationBound(core_index, int(membrane_init), int(membrane_init))
    active = matrix[:, :used_neurons]
    positive = np.maximum(active, 0).sum(axis=0)
    negative = np.minimum(active, 0).sum(axis=0)
    highest = int(theta - 1 + int(positive.max()))
    floor = int(membrane_init) + int(cycles) * int(negative.min())
    lowest = min(int(membrane_init), floor)
    return SaturationBound(core_index, lowest, highest)


def require_no_saturation(
    mapping: Any, *, spec: CoreSpec, thetas: Dict[int, int], cycles: int,
    membrane_init: int,
) -> Tuple[SaturationBound, ...]:
    """The gate the sync-fire law's contract rests on; inert for every other law.

    A law that saturates BY DESIGN (the stock unsigned register) is not judged
    here: its clamp is the modelled substrate, not a broken proof.
    """
    if not spec.asserts_no_saturation:
        return ()
    bounds: List[SaturationBound] = []
    for core_index, core in enumerate(mapping.cores):
        bound = core_saturation_bound(
            core, spec=spec, core_index=core_index,
            theta=int(thetas[core_index]), cycles=int(cycles),
            membrane_init=int(membrane_init),
        )
        if not bound.inside(spec):
            raise OdinFeasibilityError(
                KEY_NO_SATURATION,
                f"core {core_index}: the membrane provably reaches "
                f"[{bound.lowest}, {bound.highest}] over {cycles} cycles, which "
                f"is not strictly inside the declared "
                f"{spec.membrane_bits}-bit signed register "
                f"[{spec.membrane_low}, {spec.membrane_high}]. This law claims "
                f"to hold exactly the number the unbounded accumulator holds, "
                f"and that claim is false at a rail — so the deployment is "
                f"refused rather than clamped. Widen membrane_bits, lower the "
                f"fan-in, or raise theta.")
        bounds.append(bound)
    return tuple(bounds)


def _as_integer(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    snapped = int(round(number))
    return snapped if abs(number - snapped) <= 1e-9 else None
