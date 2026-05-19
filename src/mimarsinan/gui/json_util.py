"""JSON-serializable conversion for GUI payloads."""

from __future__ import annotations

from typing import Any


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except ValueError:
            pass
    return str(obj)
