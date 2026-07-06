"""Fill per-cycle signal tensors from compressed source spans."""

from __future__ import annotations

from typing import Callable, Iterable, Protocol, Sequence

import torch


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


class _GatherGroup:
    """One source's spans, collapsed to a slice (single span) or index copy."""

    __slots__ = ("src_core", "slices", "dst_index", "src_index")

    def __init__(self, src_core: int, spans: list, device) -> None:
        self.src_core = int(src_core)
        if len(spans) == 1:
            sp = spans[0]
            self.slices = (
                int(sp.dst_start), int(sp.dst_end),
                int(sp.src_start), int(sp.src_end),
            )
            self.dst_index = None
            self.src_index = None
            return
        self.slices = None
        dst = [i for sp in spans for i in range(int(sp.dst_start), int(sp.dst_end))]
        src = [i for sp in spans for i in range(int(sp.src_start), int(sp.src_end))]
        self.dst_index = torch.tensor(dst, dtype=torch.long, device=device)
        self.src_index = torch.tensor(src, dtype=torch.long, device=device)

    def copy_from(self, out, source) -> None:
        if self.slices is not None:
            d0, d1, s0, s1 = self.slices
            out[:, d0:d1] = source[:, s0:s1]
            return
        out.index_copy_(1, self.dst_index, source.index_select(1, self.src_index))


class SpanFillPlan:
    """Per-core precomputed gather plan over a STATIC span list (W3b).

    Groups the RLE spans by source so one fill is a handful of index copies
    instead of one Python slice assignment per span; bit-equal to
    :func:`fill_signal_from_spans` by construction (spans never overlap).
    """

    def __init__(self, spans: Sequence[_SpanLike], device) -> None:
        on_spans: list = []
        by_source: dict = {}
        for sp in spans:
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                on_spans.append(sp)
                continue
            key = -2 if sp.kind == "input" else int(sp.src_core)
            by_source.setdefault(key, []).append(sp)

        self._on_slice = None
        self._on_index = None
        if len(on_spans) == 1:
            self._on_slice = (int(on_spans[0].dst_start), int(on_spans[0].dst_end))
        elif on_spans:
            dst = [
                i for sp in on_spans
                for i in range(int(sp.dst_start), int(sp.dst_end))
            ]
            self._on_index = torch.tensor(dst, dtype=torch.long, device=device)

        self._input_group = None
        self._core_groups: list = []
        for key, group_spans in by_source.items():
            group = _GatherGroup(key, group_spans, device)
            if key == -2:
                self._input_group = group
            else:
                self._core_groups.append(group)

    def apply(self, out, *, input_spikes, buffers, on_value: float) -> None:
        """Zero ``out`` then fill every span group for one cycle."""
        out.zero_()
        if self._on_slice is not None:
            d0, d1 = self._on_slice
            out[:, d0:d1].fill_(float(on_value))
        elif self._on_index is not None:
            out.index_fill_(1, self._on_index, float(on_value))
        if self._input_group is not None:
            self._input_group.copy_from(out, input_spikes)
        for group in self._core_groups:
            group.copy_from(out, buffers[group.src_core])

    def tensors(self) -> Iterable[torch.Tensor]:
        """Index tensors held by this plan (cache byte accounting)."""
        if self._on_index is not None:
            yield self._on_index
        groups = list(self._core_groups)
        if self._input_group is not None:
            groups.append(self._input_group)
        for group in groups:
            if group.dst_index is not None:
                yield group.dst_index
                yield group.src_index


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
