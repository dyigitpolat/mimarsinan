"""Backend introspection tools for MimarsinanLayoutBackend.

One tool per payload the CANDIDATE view can answer, declared from the
introspection registry itself: a payload registered there reaches the agent
without a hand-written tool, and one that a candidate cannot answer is never
advertised. The agent-facing names of the original tools are pinned in
``_TOOL_NAMES`` (renaming a tool would break every saved trace).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from compilagent import ToolDecl
from pydantic import BaseModel, Field

from mimarsinan.deployment_record.introspection import (
    CANDIDATE_LAYOUT,
    INTROSPECTION_REGISTRY,
)

if TYPE_CHECKING:
    from .backend import MimarsinanLayoutBackend

# Payload name -> the agent-facing tool name, where they differ.
_TOOL_NAMES = {
    "layer_rollup": "inspect_layer_breakdown",
    "bank_composition": "inspect_weight_banks",
}

# Payloads whose tool also carries the candidate's raw objective values.
_WITH_OBJECTIVES = frozenset({"layout_stats"})


class _CandidateOnlyArgs(BaseModel):
    candidate_id: str = Field(
        description=(
            "Candidate id returned by `propose_candidate(...)`. Use "
            "`compare_runs()` to discover the ids of judged candidates."
        )
    )


class _NoArgs(BaseModel):
    pass


def tool_name_for(payload_name: str) -> str:
    return _TOOL_NAMES.get(payload_name, f"inspect_{payload_name}")


def _counted(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Row tables are long; give the agent their size next to them."""
    counts = {
        f"{key}_count": len(value)
        for key, value in envelope.items()
        if isinstance(value, list)
    }
    return {**envelope, **counts}


def build_introspection_tools(
    backend: "MimarsinanLayoutBackend",
) -> Sequence[ToolDecl]:
    """One ``ToolDecl`` per candidate-answerable payload, plus the objective catalogue."""

    def _payload(candidate_id: str) -> dict:
        try:
            return backend.get_candidate_payload(candidate_id)
        except KeyError as exc:
            known = list(backend.known_candidate_ids())[-5:]
            raise ValueError(
                f"unknown candidate `{candidate_id}` (no compile payload "
                f"cached); recently compiled: {known}"
            ) from exc

    def _make_handler(payload_name: str):
        def handler(*, candidate_id: str) -> str:
            payload = _payload(candidate_id)
            envelope = (payload.get("introspection") or {}).get(payload_name)
            body: Dict[str, Any] = {"candidate_id": candidate_id}
            if envelope is None:
                body["unavailable"] = (
                    f"payload {payload_name!r} was not served for this candidate "
                    f"(its layout could not be computed)"
                )
            else:
                body.update(_counted(dict(envelope)))
            if payload_name in _WITH_OBJECTIVES:
                body["hw_objectives"] = payload.get("hw_objectives", {})
            return json.dumps(body, indent=2, default=str)

        return handler

    def list_objectives() -> str:
        """Return the active objective catalogue with goal directions."""
        for cid in reversed(backend.known_candidate_ids()):
            payload = backend.get_candidate_payload(cid)
            catalog = payload.get("objective_catalog", [])
            if catalog:
                return json.dumps({"objectives": catalog}, indent=2, default=str)
        return json.dumps({"objectives": []}, indent=2, default=str)

    decls: List[ToolDecl] = [
        ToolDecl(
            name=tool_name_for(spec.name),
            description=(
                f"{spec.doc} Payload `{spec.name}` v{spec.version}; read-only."
            ),
            args_schema=_CandidateOnlyArgs.model_json_schema(),
            handler=_make_handler(spec.name),
            args_model=_CandidateOnlyArgs,
            read_only=True,
        )
        for spec in INTROSPECTION_REGISTRY.all()
        if CANDIDATE_LAYOUT in spec.builders
    ]
    decls.append(
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
        )
    )
    return tuple(decls)


__all__ = ["build_introspection_tools", "tool_name_for"]
