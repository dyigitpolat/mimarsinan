"""[U-LIF2 r2] The suite's point of a mode: the point a test states for a mode it names, with the window firing every
executor of the set export runs (fire while integrating, ``HISTORICAL_WINDOW_FIRING``) and no execution extension, never
the declared default. The toolchain holds no mode-only point builder (its capability judgments and policy builds take the
run's resolved point, the ratchet t02c); a test that exercises one mode's behaviour names that mode's point here."""

from __future__ import annotations

from typing import Any, Optional

from mimarsinan.chip_simulation.activation_axes import axes_from_legacy
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_point import HISTORICAL_WINDOW_FIRING, SpikingPoint, stated_point


def point_of_mode(spiking_mode: str, schedule: Any = None, spiking_variant: Any = None, *,
                  soma_law: Optional[SomaLaw] = None) -> SpikingPoint:
    """The stated point of ``spiking_mode`` (a bare ``lif`` is the synchronized variant, its historical meaning): the
    schedule and the variant a test names, the soma law it holds (else the default law), the historical window firing
    where the axis applies."""
    record: dict = {"spiking_mode": spiking_mode, "ttfs_cycle_schedule": schedule}
    if spiking_variant is not None:
        record.update(spiking_family=axes_from_legacy(spiking_mode, schedule)[0], spiking_variant=spiking_variant)
    return stated_point(record, window_firing=HISTORICAL_WINDOW_FIRING, soma_law=soma_law)


def declaring_fire_while_integrating(document: Any) -> Any:
    """[U-LIF2 W16] A test document silent on the window firing that exercises firing during the window declares it,
    as the toolchain's own windowed cells do (the owner's option A: integrate then emit is the default of a silent
    document). The key is added only where the axis applies (``window_firing_applies``), and a declaration stands."""
    from mimarsinan.chip_simulation.window_firing import window_firing_applies
    from mimarsinan.chip_simulation.window_firing_axis import FIRE_WHILE_INTEGRATING, WINDOW_FIRING_KEY

    if WINDOW_FIRING_KEY in document or not window_firing_applies(document):
        return document
    return {**document, WINDOW_FIRING_KEY: FIRE_WHILE_INTEGRATING}


def settled_window_firing(document: Any) -> Any:
    """[U-LIF2 W16] A test that moves a declaring document off the (lif, synchronized) point drops its window firing (a
    document away from the axis declares none, as the toolchain's own documents do)."""
    from mimarsinan.chip_simulation.window_firing import window_firing_applies
    from mimarsinan.chip_simulation.window_firing_axis import WINDOW_FIRING_KEY

    if WINDOW_FIRING_KEY in document and not window_firing_applies(document):
        return {k: v for k, v in document.items() if k != WINDOW_FIRING_KEY}
    return document
