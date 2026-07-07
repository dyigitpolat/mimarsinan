"""A6 install-resolution gauge cards from [MBH-A6] events."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

_GAUGE_ORDER = ("value", "temporal", "chain")


def build_a6_gauges(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The latest per-(gauge, context) verdict cards, in gauge order.

    Gauges are warn-only pre-flight instruments; the card keeps the raw
    payload so the panel can show the numbers next to the verdict.
    """
    latest: Dict[tuple, Dict[str, Any]] = {}
    for record in events:
        if record.get("kind") != "mbh_a6":
            continue
        payload = dict(record.get("payload") or {})
        gauge = str(payload.get("gauge", ""))
        context = str(payload.get("context", ""))
        latest[(gauge, context)] = {
            "gauge": gauge,
            "context": context,
            "verdict": payload.get("verdict"),
            "step": record.get("step"),
            "detail": {
                key: value for key, value in payload.items()
                if key not in ("gauge", "context", "verdict")
            },
        }
    cards: List[Dict[str, Any]] = sorted(
        latest.values(),
        key=lambda card: (
            _GAUGE_ORDER.index(card["gauge"]) if card["gauge"] in _GAUGE_ORDER else 99,
            card["context"],
        ),
    )
    return {"cards": cards}
