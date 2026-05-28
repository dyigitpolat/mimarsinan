"""Strict-by-default forward probe for conversion warmup / equivalence checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


_NODE_NAME_RE = re.compile(
    r"\[ModelRepresentation\] forward failed at node "
    r"[A-Za-z_][A-Za-z0-9_]*\(name=(?P<quote>['\"])(?P<name>[^'\"]+)(?P=quote)\)"
)


class ConversionProbeError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    ok: bool
    error: Optional[BaseException] = None
    failing_node_name: Optional[str] = None
    context: str = ""

    def format(self) -> str:
        if self.ok:
            return f"[{self.context}] probe ok"
        node_part = (
            f" at node {self.failing_node_name!r}"
            if self.failing_node_name
            else ""
        )
        return (
            f"[{self.context}] forward probe failed{node_part}: "
            f"{type(self.error).__name__}: {self.error}"
        )


def probe_forward(
    flow: nn.Module,
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str],
    *,
    batch: int = 1,
    strict: bool = True,
    context: str = "probe_forward",
) -> ProbeResult:
    """Run one zeros-tensor forward; ``strict=True`` re-raises as ``ConversionProbeError``."""
    device = torch.device(device)
    was_training = flow.training
    flow.eval()
    try:
        dummy = torch.zeros((batch, *tuple(input_shape)), device=device)
        with torch.no_grad():
            flow(dummy)
    except Exception as exc:
        failing = _extract_failing_node_name(exc)
        result = ProbeResult(
            ok=False, error=exc, failing_node_name=failing, context=context,
        )
        if strict:
            raise ConversionProbeError(result.format()) from exc
        return result
    finally:
        if was_training:
            flow.train()
    return ProbeResult(ok=True, context=context)


def _extract_failing_node_name(exc: BaseException) -> Optional[str]:
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        match = _NODE_NAME_RE.search(str(cur))
        if match is not None:
            return match.group("name")
        cur = cur.__cause__ or cur.__context__
    return None
