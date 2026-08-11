"""Layout payload collection for MimarsinanLayoutBackend.

Everything the agent may see about a candidate comes from the introspection
registry — a declared, versioned surface — so this module reads no ``mapping``
internals and cannot recompute a layout answer with a partial capability set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mimarsinan.deployment_record.introspection import (
    INTROSPECTION_REGISTRY,
    CandidateLayoutView,
    channel_envelope,
)
from mimarsinan.search.problems.joint.problem import json_key

logger = logging.getLogger(__name__)


def collect_layout_payload(
    problem: Any,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    """Serve every introspection payload the candidate can answer, plus objectives."""
    pcfg = configuration.get("platform_constraints", {})

    cache = getattr(problem, "_hw_only_cache", None)
    softcores: List[Any]
    host_segments: int
    total_params: float
    if cache is not None and getattr(problem, "search_mode", "joint") == "hardware":
        softcores = list(cache.softcores)
        host_segments = int(cache.host_side_segment_count)
        total_params = float(cache.total_params)
    else:
        mc = configuration.get("model_config", {})
        try:
            model, total_params = problem._build_model(mc, pcfg)
        except Exception:
            key = json_key(configuration)
            vc = getattr(problem, "_validation_cache", {}).get(key)
            if vc is None:
                raise
            logger.warning(
                "Model rebuild failed for candidate %.500s; serving degraded "
                "layout payload (hw_objectives only) from validation cache",
                key, exc_info=True,
            )
            return degraded_payload(dict(vc.hw_objectives))
        softcores, host_segments = problem._collect_softcores(model, pcfg)

    view = CandidateLayoutView.from_platform(
        softcores, pcfg,
        host_side_segment_count=host_segments,
        total_params=total_params,
    )
    hw_objectives, _ = problem._compute_hw_objectives(
        softcores, pcfg, total_params, host_segments,
    )
    return with_legacy_projection(
        {
            "introspection": INTROSPECTION_REGISTRY.serve_all_dicts(view),
            "hw_objectives": dict(hw_objectives or {}),
        }
    )


def degraded_payload(hw_objectives: Dict[str, Any]) -> Dict[str, Any]:
    """What a candidate whose model could not be rebuilt can still honestly say."""
    return with_legacy_projection(
        {"introspection": {}, "hw_objectives": dict(hw_objectives)}
    )


def rows_of(payload: Dict[str, Any], name: str, field: str) -> List[Dict[str, Any]]:
    """One payload's row table from a served envelope map (empty when unserved)."""
    envelope = (payload.get("introspection") or {}).get(name) or {}
    return list(envelope.get(field) or [])


def with_legacy_projection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add the flat keys the run summaries and guidance blocks read.

    A PROJECTION of the served payloads, never a second computation — the
    registry envelopes stay the source of truth (and carry their versions).
    """
    stats = (payload.get("introspection") or {}).get("layout_stats") or {}
    softcores = rows_of(payload, "softcores", "softcores")
    return {
        **payload,
        "softcore_count": len(softcores),
        "per_softcore": softcores,
        "per_layer": rows_of(payload, "layer_rollup", "layers"),
        "layout_stats": dict(stats.get("stats") or {}),
    }


# Artifact file name -> what it holds, as one rule per file (configuration,
# payload) -> body. ``introspection.json`` is the versioned one: a stored file
# outlives the process that wrote it, so it carries the CHANNEL's format version
# around the served payloads (each of which still carries its own).
_ARTIFACTS = (
    ("config.json", lambda configuration, payload: configuration),
    ("softcores.json", lambda configuration, payload: payload["per_softcore"]),
    ("layout_stats.json", lambda configuration, payload: payload["layout_stats"]),
    ("introspection.json",
     lambda configuration, payload: channel_envelope(payload["introspection"])),
)


def write_layout_artifacts(
    artifact_dir: Path, configuration: Dict[str, Any], payload: Dict[str, Any],
) -> Tuple[Path, ...]:
    """Write the candidate's artifacts; ``introspection.json`` is the versioned one."""
    written = []
    for name, body_of in _ARTIFACTS:
        path = artifact_dir / name
        path.write_text(json.dumps(body_of(configuration, payload), indent=2, default=str))
        written.append(path)
    return tuple(written)
