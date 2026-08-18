"""The ONE model-call path AgentEvolve and Compilagent-style drivers share.

Formatting lives in ``trace_format.py``; what stays here is the call itself —
make the request, count what it spent, and show it to the live monitor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, get_origin

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.optimizers.llm.trace_format import (
    TRACE_MAX_GEN_COMPLETE_STR,
    TRACE_MAX_RESPONSE_STR,
    TRACE_MAX_SECTION_CHARS,
    TRACE_MAX_SECTIONS,
    TRACE_MAX_TEXT_PREVIEW,
    coerce_llm_text,
    parse_json_object,
    schema_has_dict_type,
    split_prompt_for_trace,
    trace_response_summary,
)
from mimarsinan.search.optimizers.llm.usage import LlmUsageAccumulator
from mimarsinan.search.optimizers.search_events import emit_search_event

__all__ = [
    "LLMTraceMixin",
    "coerce_llm_text",
    "emit_search_event",
    "parse_json_object",
    "schema_has_dict_type",
    "split_prompt_for_trace",
    "trace_response_summary",
]


class LLMTraceMixin:
    """Trace emission and pydantic-ai LLM invocation."""

    model: str
    llm_retries: int
    verbose: bool
    _trace_gen: int
    _trace_seq: int
    #: [TS3] The run's usage accumulator — one per search, set by the host
    #: before its first call, so retries and regeneration rounds add up.
    _llm_usage: LlmUsageAccumulator

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

    async def _run_agent(self, agent: Any, prompt: str, output_type: Any) -> Any:
        """[TS3] THE model request: one call path, one usage capture.

        pydantic-ai accumulates into the usage object it is handed, so the
        retries INSIDE a run are counted request by request, and a run that
        ends in an exception still reports the tokens it spent — a count that
        shrank exactly when a model misbehaved would be worthless.
        """
        from pydantic_ai.usage import RunUsage

        run_usage = RunUsage()
        try:
            return await agent.run(prompt, output_type=output_type, usage=run_usage)
        finally:
            self._llm_usage.record_run_usage(run_usage)

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
                result = await self._run_agent(agent, augmented, str)
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
            result = await self._run_agent(agent, template, output_model)
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
