"""[TS3] What a search asked of a model, added up where the requests are made.

TS1's ``LlmUsage`` is the sealed FACT; this is the instrument that reaches it.
One accumulator per run, shared by every call the driver makes, so a retry
inside a run, a regeneration round after a failed batch, and a continuation of
an agent session all land in the same totals — and so a driver cannot report a
number that quietly excludes its own recovery paths.

Two report shapes reach it, and both funnel into one piece of arithmetic: a
pydantic-ai run's own usage object (the driver holds the agent), and the
per-run usage a harness reports when the agent loop belongs to someone else.
Money is deliberately absent — dollars are priced research-side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from mimarsinan.search.optimizers.budget import LlmUsage

#: The token-direction keys a harness reports its per-run usage under, and the
#: key that names its own model-call count when it keeps one.
REPORT_TOKENS_IN = "request_tokens"
REPORT_TOKENS_OUT = "response_tokens"
REPORT_CALLS = "llm_calls"


def model_name(model: Any) -> str:
    """The name a model answers to — its own, or the id string it was declared as.

    An injected model OBJECT must not seal its ``repr`` into an artifact: that
    is neither a price-table key nor stable across runs.
    """
    return str(getattr(model, "model_name", None) or model)


@dataclass
class LlmUsageAccumulator:
    """Running totals for one search, sealed into TS1's frozen ``LlmUsage``."""

    model: str
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def record(self, *, calls: int, tokens_in: int, tokens_out: int) -> None:
        """THE arithmetic — every capture path funnels through here."""
        self.calls += int(calls)
        self.tokens_in += int(tokens_in)
        self.tokens_out += int(tokens_out)

    def record_run_usage(self, usage: Any) -> None:
        """One pydantic-ai run's usage: every request it made, retries included."""
        self.record(
            calls=int(getattr(usage, "requests", 0) or 0),
            tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def record_report(self, extra: Optional[Mapping[str, Any]]) -> None:
        """One harness run's usage report, when it carried one.

        A harness that counts its own model calls names them; one that reports
        only the run's token totals counts as the single exchange we can see.
        """
        usage = (extra or {}).get("usage")
        if not isinstance(usage, Mapping):
            return
        self.record(
            calls=int((extra or {}).get(REPORT_CALLS, 1) or 1),
            tokens_in=int(usage.get(REPORT_TOKENS_IN, 0) or 0),
            tokens_out=int(usage.get(REPORT_TOKENS_OUT, 0) or 0),
        )

    def sealed(self) -> LlmUsage:
        """The frozen fact a ledger carries."""
        return LlmUsage(
            model=self.model,
            calls=self.calls,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
        )


__all__ = ["LlmUsageAccumulator", "model_name"]
