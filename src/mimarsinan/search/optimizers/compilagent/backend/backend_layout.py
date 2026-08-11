"""Layout payload collection for MimarsinanLayoutBackend.

Two rules meet here, and both are load-bearing.

WHERE the facts come from: ``problem.candidate_layout(configuration)`` — the
problem's own public seam, which walks the very path an evaluation walks. The
chip is the one the deployment resolver builds for the candidate, the softcores
are the ones the search packs, the census is the one the search scores. This
module reaches into no private member and recomputes no layout, so the agent
cannot be shown a chip or a number the search does not use.

HOW they are served: the introspection registry — a declared, versioned surface
— so this module reads no ``mapping`` internals and cannot answer with a partial
capability set. The flat keys below are a PROJECTION of the served envelopes,
never a second computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mimarsinan.deployment_record.introspection import (
    INTROSPECTION_REGISTRY,
    CandidateLayoutView,
    channel_envelope,
)
from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.search.problems.joint.types import CandidateLayout


def introspection_view(layout: CandidateLayout) -> CandidateLayoutView:
    """A ``CandidateLayout`` as the channel's view — the packing handed over, not redone.

    The problem already ran the layout; a second run is a second answer, so
    ``CandidateLayoutView.packed`` takes the census it produced.
    """
    return CandidateLayoutView.packed(
        layout.softcores,
        layout.platform,
        layout=layout.stats,
        host_side_segment_count=layout.host_side_segment_count,
        total_params=layout.view.total_params,
    )


def collect_layout_payload(
    problem: Any,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    """Serve every introspection payload the candidate can answer, plus objectives."""
    layout = problem.candidate_layout(configuration)
    view = introspection_view(layout)
    return with_legacy_projection(
        {
            "introspection": INTROSPECTION_REGISTRY.serve_all_dicts(view),
            # Every axis this candidate's STATIC facts can answer — the registry
            # read of the very view the search scores. The training proxy is not
            # among them, by construction of the candidate view.
            "hw_objectives": OBJECTIVES.extract(layout.view),
        }
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
