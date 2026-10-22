"""Strict-by-default forward probe for conversion warmup / equivalence checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal, refusal_in_chain


_NODE_NAME_RE = re.compile(
    r"\[ModelRepresentation\] forward failed at node "
    r"[A-Za-z_][A-Za-z0-9_]*\(name=(?P<quote>['\"])(?P<name>[^'\"]+)(?P=quote)\)"
)


_T = TypeVar("_T")


class ConversionProbeError(ToolchainRefusal, RuntimeError):
    """A conversion a probe refuses by name (the base of the value-parity refusal)."""


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
    dtype: Optional[torch.dtype] = None,
) -> ProbeResult:
    """Run one zeros-tensor forward; ``strict=True`` re-raises the failure.

    A named refusal raised inside the forward reaches the caller as that refusal; an
    anonymous crash as itself (loud), never wrapped in a refusal type. ``dtype`` sets the
    probe input's dtype (a converted fp64/fp16 model needs a matching probe input);
    ``None`` keeps the torch default.
    """
    dummy = torch.zeros(
        (batch, *tuple(input_shape)), device=torch.device(device),
        dtype=dtype if dtype is not None else torch.get_default_dtype(),
    )
    _, exc = guarded_forward(flow, dummy)
    if exc is None:
        return ProbeResult(ok=True, context=context)
    result = ProbeResult(
        ok=False, error=exc, failing_node_name=extract_failing_node_name(exc),
        context=context,
    )
    if strict:
        raise refusal_in_chain(exc) or exc
    return result


def guarded(call: Callable[[], _T]) -> Tuple[Optional[_T], Optional[Exception]]:
    """``(result, None)`` or ``(None, exc)``: the one place a probe captures a failure, so
    every probe stage decides in one place (a refusal passes on by name, a crash as itself)."""
    try:
        return call(), None
    except Exception as exc:
        return None, exc


def guarded_forward(
    module: nn.Module, x: torch.Tensor,
) -> Tuple[Optional[object], Optional[Exception]]:
    """``guarded`` eval-mode, grad-free forward of ``module`` on ``x``; restores the train flag."""
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            return guarded(lambda: module(x))
    finally:
        if was_training:
            module.train()


def extract_failing_node_name(exc: BaseException) -> Optional[str]:
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        match = _NODE_NAME_RE.search(str(cur))
        if match is not None:
            return match.group("name")
        cur = cur.__cause__ or cur.__context__
    return None
