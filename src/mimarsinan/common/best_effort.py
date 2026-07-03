"""The single sanctioned log-and-degrade seam for non-critical side work."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

_DEFAULT_LOGGER = logging.getLogger("mimarsinan.best_effort")


@contextmanager
def best_effort(what: str, *, logger: logging.Logger | None = None) -> Iterator[None]:
    """Run a non-critical block; log-and-continue on failure.

    Only for telemetry/rendering side work whose failure must not kill the
    pipeline. Never wraps verification, mapping, or training logic.
    """
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        (logger or _DEFAULT_LOGGER).debug("best-effort %s failed", what, exc_info=True)
