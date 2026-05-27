"""JSON-safe FastAPI response helpers for the GUI server."""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse


def sanitize(obj: Any) -> Any:
    """Recursively make values JSON-safe (delegates to ``json_util.to_json_safe``)."""
    from mimarsinan.gui.json_util import to_json_safe

    return to_json_safe(obj)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN / Inf to ``null``."""

    def default(self, o: Any) -> Any:  # noqa: D401
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(sanitize(o))


class SafeJSONResponse(JSONResponse):
    """JSONResponse that silently converts NaN/Inf to null."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            cls=SafeJSONEncoder,
        ).encode("utf-8")
