"""Capability probes: 'could an axis EVER answer here', asked of the registry.

A probe is a question, not data — its values are placeholders whose only meaning is
"this datum exists". Two levels: the CAPABILITY probe (what a search mode can carry at
all) and the RUN probe (that, narrowed by what this run actually declared).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from mimarsinan.deployment_record.objectives.spec import LayoutStatsView
from mimarsinan.deployment_record.objectives.views import (
    CandidateStaticView,
    mode_trains_accuracy,
)
from mimarsinan.deployment_record.platform_physics.probe import probe_physics
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities import CandidateQuantityContext

#: Placeholder magnitudes: strictly positive, so no availability predicate is tripped
#: by a zero (a band touching zero cannot be inverted, and a zero census reads as
#: "no work" rather than "unknown").
_PROBE_COUNT = 1.0


@dataclass(frozen=True)
class _ProbeLayout:
    """A layout-stats stand-in whose only claim is that the fields EXIST."""

    mapped_params_pct: float = 0.0
    total_wasted_axons_pct: float = 0.0
    total_wasted_neurons_pct: float = 0.0
    fragmentation_pct: float = 0.0
    schedule_sync_count: int = 0
    total_hw_cores: int = 1
    schedule_pass_count: int = 1
    neural_segment_count: int = 1


def _probe_context() -> CandidateQuantityContext:
    """Every run declaration a candidate could carry, at placeholder magnitudes."""
    return CandidateQuantityContext(
        timesteps=1,
        activity_factor=1.0,
        weight_bits=1,
        tiles=1,
        cores_physical=1,
        neurons_physical=1,
        axons_physical=1,
        host_macs=0,
        onchip_macs=1,
        host_params=0,
        onchip_params=1,
    )


# The static facts a candidate view can hold; each one is a separate question
# ("does this candidate carry a layout?"), so an objective's need for a fact is
# answered by asking the registry, never by a hand-kept list of objective names.
CANDIDATE_FRAGMENTS: Tuple[str, ...] = (
    "layout",
    "chip_param_capacity",
    "total_params",
    "host_side_segment_count",
    "estimated_accuracy",
    "physics",
    "quantity_context",
)


def _full_candidate_probe() -> CandidateStaticView:
    """Every candidate fragment populated; the values are placeholders."""
    return CandidateStaticView(
        layout=_ProbeLayout(),
        chip_param_capacity=_PROBE_COUNT,
        total_params=_PROBE_COUNT,
        host_side_segment_count=0,
        estimated_accuracy=0.0,
        physics=probe_physics(),
        quantity_context=_probe_context(),
    )


def candidate_capability_probe(search_mode: str) -> CandidateStaticView:
    """A maximally populated candidate view: what a candidate CAN carry in this mode.

    A capability question, not data — the placeholders' only meaning is "this datum
    exists in this mode".
    """
    probe = _full_candidate_probe()
    if mode_trains_accuracy(search_mode):
        return probe
    return replace(probe, estimated_accuracy=None)


def run_capability_probe(
    search_mode: str, physics: Optional[PlatformPhysics]
) -> CandidateStaticView:
    """The capability probe narrowed by what THIS run declared.

    The mode says which axes a candidate could carry; the run's own physics says
    which of those it can actually back. With no profile declared the absolute axes
    are unavailable, so asking for one is refused BY NAME at resolution time rather
    than producing a number no vendor stands behind.
    """
    return replace(candidate_capability_probe(search_mode), physics=physics)


def candidate_probe_without(fragment: str) -> CandidateStaticView:
    """A fully populated candidate view MINUS one fragment.

    Asking which objectives go unavailable on it is how a caller learns whether a
    fragment is worth computing — the registry answers, so a new objective
    classifies itself.
    """
    if fragment not in CANDIDATE_FRAGMENTS:
        raise ValueError(
            f"unknown candidate fragment {fragment!r}; a candidate view carries "
            f"{list(CANDIDATE_FRAGMENTS)}"
        )
    return replace(_full_candidate_probe(), **{fragment: None})


_PROBE_LAYOUT_IS_A_LAYOUT: type[LayoutStatsView] = _ProbeLayout
