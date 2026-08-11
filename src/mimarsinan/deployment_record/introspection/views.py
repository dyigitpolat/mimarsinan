"""The two introspection views: a search candidate, and a sealed record.

The same two-completeness idea the objectives registry uses (schema doc §3): a
candidate holds the shape-only layout its hook already computes, a sealed run
holds every fragment. "Is this payload available" ≡ "is its backing datum
populated" — asked of the view, never of a hand-maintained list.

The candidate view is also the ONE place a consumer turns (softcores, platform)
into a layout answer, so no optimizer has to call ``mapping``'s verification
helpers itself — and none can forget the capability declaration on the way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)

CANDIDATE_LAYOUT = "candidate_layout"
DEPLOYMENT_RECORD = "deployment_record"


class IntrospectionView(Protocol):
    """What a payload builder may read; the kind selects the builder."""

    @property
    def view_kind(self) -> str: ...


def core_types_from_platform(
    platform: Mapping[str, Any]
) -> Tuple[LayoutHardCoreType, ...]:
    """The declared core grid as layout types (missing ``count`` = one core)."""
    return tuple(
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct.get("count", 1)),
        )
        for ct in (platform.get("cores") or ())
    )


@dataclass(frozen=True)
class CandidateLayoutView:
    """A search candidate's layout: shape-only softcores against a declared chip."""

    softcores: Tuple[LayoutSoftCoreSpec, ...]
    capabilities: ChipCapabilities
    core_types: Tuple[LayoutHardCoreType, ...] = ()
    layout: Optional[LayoutVerificationStats] = None
    layout_error: Optional[str] = None
    host_side_segment_count: Optional[int] = None
    total_params: Optional[float] = None
    platform: Mapping[str, Any] = field(default_factory=dict)

    @property
    def view_kind(self) -> str:
        return CANDIDATE_LAYOUT

    @classmethod
    def from_platform(
        cls,
        softcores: Sequence[LayoutSoftCoreSpec],
        platform: Mapping[str, Any],
        *,
        host_side_segment_count: Optional[int] = None,
        total_params: Optional[float] = None,
    ) -> "CandidateLayoutView":
        """Lay the candidate out exactly as deployment would read the platform.

        The capability declaration is forwarded WHOLE (``layout_kwargs``), so the
        pass structure this view reports is the one the builder will compose.
        """
        capabilities = ChipCapabilities.from_platform_constraints(platform)
        core_types = core_types_from_platform(platform)
        stats: Optional[LayoutVerificationStats] = None
        error: Optional[str] = None
        if softcores and core_types:
            stats, error = compute_mapping_stats(
                softcores=list(softcores),
                core_types=list(core_types),
                **capabilities.layout_kwargs(),
            )
        return cls(
            softcores=tuple(softcores),
            capabilities=capabilities,
            core_types=core_types,
            layout=stats,
            layout_error=error,
            host_side_segment_count=host_side_segment_count,
            total_params=total_params,
            platform=dict(platform),
        )


@dataclass(frozen=True)
class RecordIntrospectionView:
    """A sealed deployment record as an introspection view."""

    record: DeploymentRecord

    @property
    def view_kind(self) -> str:
        return DEPLOYMENT_RECORD

    @property
    def platform(self) -> Mapping[str, Any]:
        return self.record.identity.platform

    @property
    def deployment_options(self) -> Mapping[str, Any]:
        return self.record.identity.deployment_options


def capability_bits_of(view: Any) -> Optional[Mapping[str, Any]]:
    """The COMPLETE capability declaration behind a view, or ``None``.

    A candidate carries the resolved object; a sealed record carries the
    verbatim resolved platform, which is what the object was built from — the
    same declaration, read back.
    """
    capabilities = getattr(view, "capabilities", None)
    if isinstance(capabilities, ChipCapabilities):
        return capabilities.capability_bits()
    platform = getattr(view, "platform", None)
    if platform:
        return ChipCapabilities.from_platform_constraints(platform).capability_bits()
    return None
