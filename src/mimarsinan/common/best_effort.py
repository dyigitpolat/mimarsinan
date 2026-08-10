"""The single sanctioned log-and-degrade seam for non-critical side work."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

_DEFAULT_LOGGER = logging.getLogger("mimarsinan.best_effort")


@dataclass
class BestEffortOutcome:
    """What a ``best_effort`` block swallowed: the exception, or ``None`` on success."""

    error: Exception | None = None


@contextmanager
def best_effort(what: str, *, logger: logging.Logger | None = None) -> Iterator[BestEffortOutcome]:
    """Run a non-critical block; log-and-continue on failure.

    Only for telemetry/rendering side work whose failure must not kill the
    pipeline. Never wraps verification, mapping, or training logic. Yields a
    ``BestEffortOutcome`` whose ``error`` carries any swallowed exception so
    degrade paths can name the failure they are degrading from.
    """
    outcome = BestEffortOutcome()
    try:
        yield outcome
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        outcome.error = exc
        (logger or _DEFAULT_LOGGER).debug("best-effort %s failed", what, exc_info=True)
