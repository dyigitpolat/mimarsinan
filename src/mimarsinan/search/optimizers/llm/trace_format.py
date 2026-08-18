"""How an LLM exchange is SHOWN: prompt sections and response summaries.

Pure formatting for the live monitor, kept apart from the call path in
``trace.py`` so a driver's trace vocabulary can grow without touching the one
place a model request is made.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, get_args, get_origin

TRACE_MAX_SECTION_CHARS = 4000
TRACE_MAX_SECTIONS = 12
TRACE_MAX_RESPONSE_STR = 2500
TRACE_MAX_TEXT_PREVIEW = 12000
TRACE_MAX_GEN_COMPLETE_STR = 10000


def coerce_llm_text(val: Any) -> str:
    """Normalize LLM fields expected to be str; models sometimes return dict/list."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except TypeError:
        return str(val)


def parse_json_object(raw: str) -> Any:
    """Parse a JSON object out of raw LLM text; degrade to {} on malformed JSON."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return {}
    return {}


def schema_has_dict_type(output_schema: Dict[str, type]) -> bool:
    """Return True if any field type contains an open dict (additionalProperties issue)."""
    for field_type in output_schema.values():
        origin = get_origin(field_type)
        if origin is dict:
            return True
        if origin is list:
            args = get_args(field_type)
            if args and get_origin(args[0]) is dict:
                return True
    return False


def split_prompt_for_trace(prompt_text: str) -> Tuple[List[Dict[str, str]], bool, int]:
    """Split prompt into labeled sections for GUI; return (sections, truncated, total_chars)."""
    total_chars = len(prompt_text)
    parts = re.split(r"\n\s*\n+", prompt_text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    truncated = len(parts) > TRACE_MAX_SECTIONS
    sections: List[Dict[str, str]] = []
    for i, p in enumerate(parts[:TRACE_MAX_SECTIONS]):
        if len(p) > TRACE_MAX_SECTION_CHARS:
            p = p[:TRACE_MAX_SECTION_CHARS] + "\n…"
            truncated = True
        first_line = p.split("\n", 1)[0].strip()
        label = first_line[:72] + ("…" if len(first_line) > 72 else "")
        if len(label) < 8:
            label = f"Section {i + 1}"
        sections.append({"label": label, "text": p})
    return sections, truncated, total_chars


def trace_response_summary(
    call_kind: str,
    result: Any,
    *,
    prettify_configuration,
) -> Dict[str, Any]:
    """Structured response summary for live GUI (not raw JSON dumps)."""
    if hasattr(result, "model_dump"):
        result = SimpleNamespace(**result.model_dump())
    out: Dict[str, Any] = {"call_kind": call_kind}

    def _preview_cfg(d: Dict[str, Any]) -> str:
        s = prettify_configuration(d) if isinstance(d, dict) else str(d)
        return s[:280] + ("…" if len(s) > 280 else "")

    if call_kind in (
        "initial_candidates",
        "regenerate_candidates",
        "offspring",
        "regenerate_offspring",
    ):
        reasoning = coerce_llm_text(getattr(result, "reasoning", "") or "")
        cands = getattr(result, "candidates", []) or []
        out["reasoning_preview"] = reasoning[:TRACE_MAX_RESPONSE_STR] + (
            "…" if len(reasoning) > TRACE_MAX_RESPONSE_STR else ""
        )
        out["candidate_count"] = len(cands)
        previews = []
        for i, c in enumerate(cands[:2]):
            if isinstance(c, dict):
                previews.append({"index": i, "summary": _preview_cfg(c)})
            elif isinstance(c, str):
                previews.append({"index": i, "summary": c[:280]})
        out["candidate_previews"] = previews
        return out

    if call_kind == "failure_insights":
        insights = getattr(result, "insights", []) or []
        items = []
        for i, s in enumerate(insights[:20]):
            t = str(s)
            items.append({"index": i, "text": t[:400] + ("…" if len(t) > 400 else "")})
        out["insight_count"] = len(insights)
        out["insights"] = items
        return out

    text_kinds = (
        ("constraint_instruction", "constraint_instruction"),
        ("update_constraint", "updated_instruction"),
        ("performance_insights", "performance_insights"),
        ("update_performance_insights", "updated_insights"),
    )
    for kind, attr in text_kinds:
        if call_kind == kind:
            text = coerce_llm_text(getattr(result, attr, "") or "")
            cap = TRACE_MAX_TEXT_PREVIEW
            out["text_preview"] = text[:cap] + ("…" if len(text) > cap else "")
            out["text_preview_truncated"] = len(text) > cap
            out["text_preview_full_len"] = len(text)
            return out

    out["note"] = "unrecognized call_kind for trace"
    return out


__all__ = [
    "TRACE_MAX_GEN_COMPLETE_STR",
    "TRACE_MAX_RESPONSE_STR",
    "TRACE_MAX_SECTIONS",
    "TRACE_MAX_SECTION_CHARS",
    "TRACE_MAX_TEXT_PREVIEW",
    "coerce_llm_text",
    "parse_json_object",
    "schema_has_dict_type",
    "split_prompt_for_trace",
    "trace_response_summary",
]
