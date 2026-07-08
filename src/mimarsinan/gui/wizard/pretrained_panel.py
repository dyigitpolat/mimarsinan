"""The dedicated Pretrained-weights panel: builder-enriched legality + facts.

The pretrained regime's legality depends on the model-builder registration, which
only the wizard folds in (config_schema stays builder-agnostic). This module owns
that enrichment and the always-computable panel block the frontend renders from.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.common.pretrained import (
    applicable_weight_sets,
    derived_weight_set_id,
    preload_unavailable_reason,
    registered_weight_sets,
)
from mimarsinan.config_schema.registry import parse_deployment_document
from mimarsinan.config_schema.resolve import effective_view
from mimarsinan.config_schema.validation import legality_errors
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

# The pretrained knobs whose legality depends on the BUILDER registration.
PRETRAINED_LEGALITY_KEYS = ("preload_weights", "pretrained_weight_set")


def effective_with_builder(draft: Dict[str, Any]) -> Dict[str, Any]:
    """The always-computable effective config with the model builder's
    registrations folded in — so the pretrained legal sets see the builder's
    weight sets even while the draft errors (like the vehicle rows)."""
    dp = parse_deployment_document(draft or {}).dp
    cfg = effective_view(dp)
    profile = ModelRegistry.get_workload_profile(str(cfg.get("model_type") or ""))
    if profile is not None:
        for key, value in profile.config_updates().items():
            cfg.setdefault(key, value)
    return cfg


def pretrained_legality_errors(
    effective: Dict[str, Any], draft: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Keyed, remediable legality rows for the pretrained knobs, judged against
    the builder-enriched config (the same rule + remedies the framework uses)."""
    dp = parse_deployment_document(draft or {}).dp
    declared = {k: dp[k] for k in PRETRAINED_LEGALITY_KEYS if k in dp}
    return legality_errors(effective, declared)


def pretrained_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The dedicated panel's data — ALWAYS computable, so an unrelated draft
    error never empties it. Reveals every registered fact of the applicable
    weight sets plus the switch/selection state."""
    registered = registered_weight_sets(cfg)
    applicable = applicable_weight_sets(cfg) or ()
    legal_ids = [str(record["id"]) for record in applicable]
    if applicable:
        legal_preload: List[bool] | None = [False, True]
    elif registered is not None:
        legal_preload = [False]
    else:
        legal_preload = None
    return {
        "consulted": registered is not None,
        "available": bool(applicable),
        "reason": preload_unavailable_reason(cfg),
        "legal_preload": legal_preload,
        "legal_set_ids": legal_ids,
        "sets": list(applicable),
        "all_sets": list(registered or ()),
        "selected": derived_weight_set_id(cfg) or (legal_ids[0] if legal_ids else None),
    }
