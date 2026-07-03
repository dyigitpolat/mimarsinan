"""Shared LLM trace helpers for AgentEvolve and Compilagent optimizers."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, get_args, get_origin

from mimarsinan.common.best_effort import best_effort


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


def emit_search_event(reporter: Any, event: Dict[str, Any]) -> None:
    """Emit a structured search event via the reporter."""
    if reporter is None:
        return
    with best_effort("emit search_event"):
        reporter("search_event", json.dumps(event, default=str))


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


class LLMTraceMixin:
    """Trace emission and pydantic-ai LLM invocation."""

    model: str
    llm_retries: int
    verbose: bool
    _trace_gen: int
    _trace_seq: int

    if TYPE_CHECKING:

        def _log(self, message: str) -> None: ...

        @staticmethod
        def _report_search_event(reporter: Any, event: Dict[str, Any]) -> None: ...

    _TRACE_MAX_SECTION_CHARS = TRACE_MAX_SECTION_CHARS
    _TRACE_MAX_SECTIONS = TRACE_MAX_SECTIONS
    _TRACE_MAX_RESPONSE_STR = TRACE_MAX_RESPONSE_STR
    _TRACE_MAX_TEXT_PREVIEW = TRACE_MAX_TEXT_PREVIEW
    _TRACE_MAX_GEN_COMPLETE_STR = TRACE_MAX_GEN_COMPLETE_STR

    def _make_agent(self) -> Any:
        """Create a fresh pydantic-ai Agent per call (avoid stale async HTTP clients)."""
        from pydantic_ai import Agent

        return Agent(model=self.model, retries=self.llm_retries)

    @staticmethod
    def _coerce_llm_text(val: Any) -> str:
        return coerce_llm_text(val)

    @staticmethod
    def _schema_has_dict_type(output_schema: Dict[str, type]) -> bool:
        return schema_has_dict_type(output_schema)

    def _split_prompt_for_trace(self, prompt_text: str) -> Tuple[List[Dict[str, str]], bool, int]:
        return split_prompt_for_trace(prompt_text)

    def _trace_response_summary(self, call_kind: str, result: Any) -> Dict[str, Any]:
        from mimarsinan.search.optimizers.agent_evolve.schema import prettify_configuration

        return trace_response_summary(
            call_kind, result, prettify_configuration=prettify_configuration,
        )

    def _emit_llm_trace(
        self,
        call_kind: str,
        prompt_sent: str,
        output_schema: Dict[str, type],
        result: Any,
    ) -> None:
        """Emit one llm_trace search_event for the live monitor."""
        rep = getattr(self, "_trace_reporter", None)
        if rep is None:
            return
        self._trace_seq += 1
        sections, truncated, total_chars = self._split_prompt_for_trace(prompt_sent)
        schema_keys = list(output_schema.keys())
        with best_effort("emit llm_trace search_event"):
            self._report_search_event(rep, {
                "type": "llm_trace",
                "gen": self._trace_gen,
                "ordinal": self._trace_seq,
                "call_kind": call_kind,
                "output_schema_keys": schema_keys,
                "request": {
                    "sections": sections,
                    "truncated": truncated,
                    "total_chars": total_chars,
                },
                "response": self._trace_response_summary(call_kind, result),
            })

    async def _llm_call(
        self,
        template: str,
        output_schema: Dict[str, type],
        call_kind: str = "unknown",
    ) -> Any:
        """Make an LLM call with the given template and output schema."""
        agent = self._make_agent()

        try:
            if self._schema_has_dict_type(output_schema):
                keys = list(output_schema.keys())
                augmented = (
                    template
                    + f"\n\nRespond with a single valid JSON object containing exactly "
                    f"these keys: {keys}. Output only the JSON — no markdown, no explanation."
                )
                result = await agent.run(augmented, output_type=str)
                raw = getattr(result, "output", "") or ""

                data = parse_json_object(raw)

                ns: Dict[str, Any] = {}
                for k, v in output_schema.items():
                    val = data.get(k)
                    if val is None:
                        origin = get_origin(v)
                        ns[k] = [] if (origin is list or origin is dict) else ""
                    else:
                        ns[k] = val
                out = SimpleNamespace(**ns)
                self._emit_llm_trace(call_kind, augmented, output_schema, out)
                return out

            from pydantic import BaseModel, create_model

            field_definitions: Dict[str, Any] = {
                k: (v, ...) for k, v in output_schema.items()
            }
            output_model = create_model(
                "_OutputModel",
                __base__=BaseModel,
                **field_definitions,
            )
            result = await agent.run(template, output_type=output_model)
            out = getattr(result, "output", result)
            self._emit_llm_trace(call_kind, template, output_schema, out)
            return out
        except Exception as e:
            if self.verbose:
                self._log(f"  LLM error ({call_kind}): {e}")
                chain = e
                depth = 0
                while chain.__cause__ is not None and depth < 20:
                    chain = chain.__cause__
                    depth += 1
                    self._log(f"    cause: {chain}")
            raise
