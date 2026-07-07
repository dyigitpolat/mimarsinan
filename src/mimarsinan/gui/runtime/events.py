"""Structured pipeline events: the console [TAG] vocabulary as data (events.jsonl)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# Event kinds mirror the pipeline's console tags one-to-one; the print sites
# keep their [TAG] lines for console ergonomics and additionally emit one
# reporter.event(kind, payload) — nothing ever parses our own prints live.
EVENT_MBH_GATE = "mbh_gate"          # payload.action: entry|accept|reject|stall
EVENT_MBH_ENDPOINT = "mbh_endpoint"  # endpoint train-to-D-hat report
EVENT_MBH_HOP = "mbh_hop"            # hop-staged install re-affine
EVENT_MBH_PREFIX = "mbh_prefix"      # prefix-ramp stage re-affine
EVENT_MBH_A6 = "mbh_a6"              # install-resolution gauges (value|temporal|chain)
EVENT_LR_REFUSAL = "lr_refusal"      # destructive-LR refusal
EVENT_PROFILE = "profile"            # per-step wall + metric delta
EVENT_PARITY = "parity"              # torch<->deployed-sim agreement reads
EVENT_RETENTION = "retention"        # step retention-gate failure

KNOWN_EVENT_KINDS = frozenset({
    EVENT_MBH_GATE,
    EVENT_MBH_ENDPOINT,
    EVENT_MBH_HOP,
    EVENT_MBH_PREFIX,
    EVENT_MBH_A6,
    EVENT_LR_REFUSAL,
    EVENT_PROFILE,
    EVENT_PARITY,
    EVENT_RETENTION,
})


@dataclass(frozen=True)
class PipelineEvent:
    """One structured pipeline event (persisted to events.jsonl, WS-broadcast)."""

    seq: int
    step_name: str
    kind: str
    payload: Dict[str, Any]
    timestamp: float

    def to_record(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "step": self.step_name,
            "kind": self.kind,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
