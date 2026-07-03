"""Fill per-cycle signal tensors from compressed source spans."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence


class _SpanLike(Protocol):
    """Read-only span surface satisfied by the frozen ``SpikeSourceSpan`` dataclass."""

    @property
    def kind(self) -> str: ...
    @property
    def dst_start(self) -> int: ...
    @property
    def dst_end(self) -> int: ...
    @property
    def src_core(self) -> int: ...
    @property
    def src_start(self) -> int: ...
    @property
    def src_end(self) -> int: ...


def fill_signal_from_spans(
    out,
    spans: Sequence[_SpanLike],
    *,
    read_input: Callable[[_SpanLike], None],
    read_upstream: Callable[[_SpanLike], None],
    on_always_on: Callable[[int, int], None] | None = None,
    cycle: int = 0,
) -> None:
    """Zero ``out`` then fill each span slice (B, N) for the given cycle."""
    out.zero_()
    for sp in spans:
        d0 = int(sp.dst_start)
        d1 = int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            if on_always_on is not None:
                on_always_on(d0, d1)
            else:
                out[:, d0:d1].fill_(1.0)
            continue
        if sp.kind == "input":
            read_input(sp)
            continue
        read_upstream(sp)
