"""[TS3] The harness stream, read for the usage the session summary drops.

``run_session`` returns the LAST continuation's metadata, so a session that
continued three times would report a third of what it asked. Each run reports
its own totals on the event that ends it, and that stream is the only place the
WHOLE session is visible — so mimarsinan observes it on the way past.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Tuple

from compilagent import HarnessRunRequest, StreamEvent, StreamEventKind

from mimarsinan.search.optimizers.llm.usage import LlmUsageAccumulator

#: The events that END one harness run, and so the ones that report its usage.
_RUN_ENDED = (StreamEventKind.RUN_FINISHED, StreamEventKind.RUN_FAILED)


@dataclass
class UsageObservingHarness:
    """Delegates to a harness, adding up every usage report its runs emit.

    A run that FAILED is counted too: the requests it made before failing were
    made, and a total that shrank on failure would flatter the driver.
    """

    harness: Any
    usage: LlmUsageAccumulator
    # The harness protocol declares these as plain attributes, so a wrapper
    # that answers with properties is not one.
    id: str = field(init=False)
    supported_providers: Tuple[str, ...] = field(init=False)
    example_models: Tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.id = str(self.harness.id)
        self.supported_providers = tuple(self.harness.supported_providers)
        self.example_models = tuple(self.harness.example_models)

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        async for event in self.harness.run(request):
            if event.kind in _RUN_ENDED:
                self.usage.record_report(event.extra)
            yield event

    def build_continuation_request(
        self, previous: HarnessRunRequest, snapshot: Any,
    ) -> HarnessRunRequest:
        return self.harness.build_continuation_request(previous, snapshot)


__all__ = ["UsageObservingHarness"]
