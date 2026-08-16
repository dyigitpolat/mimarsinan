"""The candidate view's physics and quantity-context fragments (N0).

The axis gate (``resolve_active_specs(physics=...)``) and the view extraction
must answer from ONE source: the resolved platform the candidate decodes to.
A view built without these fragments admits an axis it cannot extract — the
N0 incident (every candidate raised at extraction while the gate said yes).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mimarsinan.deployment_record.objectives import (
    candidate_context_from_platform,
)
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.mapping.noc import LayoutNocFragments, collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    estimate_onchip_fraction,
)

#: (params, macs) estimates of the host/on-chip split — one flow walk each.
OnchipCensus = Tuple[OnchipFractionEstimate, OnchipFractionEstimate]


def candidate_fragments(pcfg: Dict, census: Optional[OnchipCensus] = None):
    """This candidate's physics and quantity context, off its OWN platform."""
    payload = pcfg.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    params_est, macs_est = census if census is not None else (None, None)
    context = candidate_context_from_platform(
        pcfg,
        host_macs=None if macs_est is None else int(macs_est.host),
        onchip_macs=None if macs_est is None else int(macs_est.onchip),
        host_params=None if params_est is None else int(params_est.host),
        onchip_params=None if params_est is None else int(params_est.onchip),
    )
    return physics, context


def collect_candidate_noc(
    *,
    softcores,
    core_types,
    census,
    pcfg: Dict,
) -> Optional[LayoutNocFragments]:
    """The candidate's NoC fragments, planned under its OWN capability bits.

    The capability declaration is forwarded WHOLE (``layout_kwargs``), the
    same discipline as the packing census — a fragment set planned under
    different scheduling permissions would place traffic on a program the
    chip never runs.
    """
    if census is None:
        return None
    return collect_noc_fragments(
        softcores=softcores,
        core_types=core_types,
        census=census,
        **ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs(),
    )


def compute_onchip_census(
    model: Any,
    input_shape: Tuple[int, ...],
    num_classes: int,
    placement: str,
) -> OnchipCensus:
    """Host/on-chip param+MAC counts through the deployment's own estimator."""
    return (
        estimate_onchip_fraction(
            model, tuple(input_shape), int(num_classes),
            encoding_placement=placement, metric="params",
        ),
        estimate_onchip_fraction(
            model, tuple(input_shape), int(num_classes),
            encoding_placement=placement, metric="macs",
        ),
    )
