"""Backend introspection tools surfaced by ``MimarsinanLayoutBackend``.

These tools let the agent ask follow-up questions about a previously
compiled candidate without re-running ``compile``: per-softcore breakdown,
per-layer aggregates, the full ``LayoutVerificationStats`` snapshot, and
the canonical objective catalogue. All four are read-only and pull from
the per-candidate payload the backend caches in ``compile``.

Each tool is wrapped in a ``ToolDecl`` with a typed Pydantic args model
so the harness adapters can validate wire-shaped JSON before calling the
handler. Handlers return JSON strings (compilagent's standard
``returns_kind="json"``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Sequence

from compilagent import ToolDecl
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .backend import MimarsinanLayoutBackend


# ---------------------------------------------------------------------- args


class _CandidateOnlyArgs(BaseModel):
    candidate_id: str = Field(
        description=(
            "Candidate id returned by `propose_candidate(...)`. Use "
            "`compare_runs()` to discover the ids of judged candidates."
        )
    )


class _NoArgs(BaseModel):
    pass


# ---------------------------------------------------------------------- impl


def build_introspection_tools(
    backend: "MimarsinanLayoutBackend",
) -> Sequence[ToolDecl]:
    """Return the four ``ToolDecl``s the backend advertises."""

    def _payload(candidate_id: str) -> dict:
        try:
            return backend.get_candidate_payload(candidate_id)
        except KeyError as exc:
            known = list(backend.known_candidate_ids())[-5:]
            raise ValueError(
                f"unknown candidate `{candidate_id}` (no compile payload "
                f"cached); recently compiled: {known}"
            ) from exc

    def inspect_softcores(*, candidate_id: str) -> str:
        """List every softcore the candidate emits, with shape + tags."""

        payload = _payload(candidate_id)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "count": len(payload.get("softcores", [])),
                "softcores": payload.get("softcores", []),
            },
            indent=2,
            default=str,
        )

    def inspect_layer_breakdown(*, candidate_id: str) -> str:
        """Per-layer aggregate (softcore count, total area, threshold groups)."""

        payload = _payload(candidate_id)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "layer_count": len(payload.get("per_layer", [])),
                "per_layer": payload.get("per_layer", []),
            },
            indent=2,
            default=str,
        )

    def inspect_layout_stats(*, candidate_id: str) -> str:
        """Full ``LayoutVerificationStats`` (utilisation, fragmentation, ...)."""

        payload = _payload(candidate_id)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "layout_stats": payload.get("layout_stats", {}),
                "hw_objectives": payload.get("hw_objectives", {}),
            },
            indent=2,
            default=str,
        )

    def list_objectives() -> str:
        """Return the active objective catalogue with goal directions."""

        # Pull from the most recently compiled candidate so the agent
        # always sees the live problem's catalogue. Falls back to an
        # empty list when no candidate has been compiled yet.
        for cid in reversed(backend.known_candidate_ids()):
            payload = backend.get_candidate_payload(cid)
            catalog = payload.get("objective_catalog", [])
            if catalog:
                return json.dumps(
                    {"objectives": catalog}, indent=2, default=str,
                )
        return json.dumps({"objectives": []}, indent=2, default=str)

    decls: List[ToolDecl] = [
        ToolDecl(
            name="inspect_softcores",
            description=(
                "Return every softcore (one logical tile of the workload) "
                "that the named candidate emits, including input/output "
                "wire counts, threshold-group id, latency tag and segment "
                "id. Read-only."
            ),
            args_schema=_CandidateOnlyArgs.model_json_schema(),
            handler=inspect_softcores,
            args_model=_CandidateOnlyArgs,
            read_only=True,
        ),
        ToolDecl(
            name="inspect_layer_breakdown",
            description=(
                "Aggregate softcores by source layer for a candidate: how "
                "many tiles each layer emits, total area, max input/output "
                "counts, latency tier counts. Read-only."
            ),
            args_schema=_CandidateOnlyArgs.model_json_schema(),
            handler=inspect_layer_breakdown,
            args_model=_CandidateOnlyArgs,
            read_only=True,
        ),
        ToolDecl(
            name="inspect_layout_stats",
            description=(
                "Return the full LayoutVerificationStats snapshot for a "
                "candidate (utilisation, fragmentation, schedule passes, "
                "wasted axons/neurons, etc.) plus the raw HW objective "
                "values. Read-only."
            ),
            args_schema=_CandidateOnlyArgs.model_json_schema(),
            handler=inspect_layout_stats,
            args_model=_CandidateOnlyArgs,
            read_only=True,
        ),
        ToolDecl(
            name="list_objectives",
            description=(
                "Return the active objective catalogue with goal "
                "directions ('min' or 'max') so the agent can reason "
                "about trade-offs across all axes. Read-only."
            ),
            args_schema=_NoArgs.model_json_schema(),
            handler=list_objectives,
            args_model=_NoArgs,
            read_only=True,
        ),
    ]
    return tuple(decls)


__all__ = ["build_introspection_tools"]
