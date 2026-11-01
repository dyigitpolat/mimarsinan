"""[U-SC3] The LIF NF walk's trains, read one cycle at a time where autograd records nothing.

A per-cycle walk reads a hop's input trains cycle by cycle (``_run_window``: ``dep_train[t]``). Built eagerly, every
boundary re-encode was the whole ``(T, B, ...)`` tensor, held twice (the 0/1 comb and its scaled copy) beside the hop's
output: S1's largest hop held 29.53 GiB at the 1,024-image validation batch (``phase_c_r/g1_tlif/s1mem``). Where no
autograd graph is recorded (every walk of the deployed composition: the AQ D-hat, the LIF finalize twin, LIF
Adaptation's metric, the parity measurement plane, the certificates' walks), a re-encode, its scaled copy, a gather
of trains and the per-cycle maps of a train (a mixed gather's multiplicities, a core's fold) are ``CycleTrain`` s:
cycle ``t`` is computed when a reader reads it, by the same elementwise or layout operation the eager walk applied to
the whole train, so every cycle is the eager train's slice bit for bit. Where autograd records, the factories return
the eager tensors unchanged (autograd saves every cycle's inputs for the backward pass, so streaming would bound
nothing, and the gradient's summation order stays the eager one).

A cycle is laid out as the eager stack lays it out (``_as_stacked``), so a consumer's kernel reads the same strides.
A re-encode's cycle is built in row blocks of its batch (at most an eighth of the rows and ``_BLOCK_ELEMENTS``
elements a block): the comb is elementwise, so the blocks are bit-equal to the whole cycle, and its float64 working
set stays a small fraction of one cycle. A mean over the cycles is taken on the materialized train (``train_mean``),
the eager reduction.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence

import torch

from mimarsinan.spiking.spike_trains import uniform_comb_cycle, uniform_comb_operands, uniform_spike_train

__all__ = [
    "CycleStack",
    "CycleTrain",
    "SegmentReaders",
    "encode_train",
    "is_event_train",
    "map_cycles",
    "map_train",
    "scale_train",
    "streams",
    "train_mean",
]

#: Elements of one row block of a re-encode's cycle (the comb's float64 working set is a few times this).
_BLOCK_ELEMENTS = 1 << 20
#: A cycle with at least this many rows is built in at least this many blocks.
_MIN_BLOCKS = 8


def streams() -> bool:
    """Whether the walk reads its trains per cycle: where autograd records nothing (module docstring)."""
    return not torch.is_grad_enabled()


def _as_stacked(cycle: torch.Tensor) -> torch.Tensor:
    """``cycle`` laid out as ``torch.stack`` lays out each slice of a train of such cycles."""
    return cycle if cycle.is_contiguous() else torch.stack([cycle], dim=0)[0]


class CycleTrain:
    """A ``(T, ...)`` train whose cycle ``t`` is computed when read (``train[t]``); its shape, dtype and device are
    one cycle's."""

    def __init__(self, T: int) -> None:
        self.T = int(T)

    def cycle(self, t: int) -> torch.Tensor:
        raise NotImplementedError

    def _meta(self) -> tuple:
        """``(cycle shape, dtype, device)``."""
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        return self._meta()[1]

    @property
    def device(self) -> torch.device:
        return self._meta()[2]

    def __getitem__(self, t) -> torch.Tensor:
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError(f"a CycleTrain is read one cycle at a time, by an integer index, not {t!r}")
        if not -self.T <= t < self.T:
            raise IndexError(f"cycle {t} of a {self.T}-cycle train")
        return self.cycle(t % self.T)

    def __len__(self) -> int:
        return self.T

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.T, *self._meta()[0]))

    def dim(self) -> int:
        return len(self._meta()[0]) + 1

    def materialize(self) -> torch.Tensor:
        """The whole train, as the eager walk built it."""
        return torch.stack([self.cycle(t) for t in range(self.T)], dim=0)

    def is_event_train(self) -> bool:
        """Every cycle a 0/1 event train (the eager check over the whole train, cycle by cycle)."""
        return all(bool(torch.all((c == 0) | (c == 1))) for c in (self.cycle(t) for t in range(self.T)))


class EncodedTrain(CycleTrain):
    """A uniform boundary re-encode of a rate (``uniform_spike_train``'s cycles, built in row blocks)."""

    def __init__(self, rate: torch.Tensor, T: int, *, phase_dither: bool = False,
                 channels: Optional[torch.Tensor] = None) -> None:
        self._rate_c, self._offsets = uniform_comb_operands(rate, T, phase_dither=phase_dither, channels=channels)
        super().__init__(T)

    def _meta(self) -> tuple:
        return self._rate_c.shape, self._rate_c.dtype, self._rate_c.device

    def _rows_per_block(self) -> int:
        rate_c = self._rate_c
        if rate_c.dim() < 2 or not rate_c.is_contiguous():
            return max(1, int(rate_c.shape[0]) if rate_c.dim() else 1)
        rows = int(rate_c.shape[0])
        per_row = max(1, rate_c[0].numel())
        return max(1, min(-(-rows // _MIN_BLOCKS), _BLOCK_ELEMENTS // per_row))

    def cycle(self, t: int) -> torch.Tensor:
        rate_c = self._rate_c
        step = self._rows_per_block()
        if rate_c.dim() < 2 or step >= int(rate_c.shape[0]):
            return _as_stacked(uniform_comb_cycle(rate_c, t, self.T, self._offsets))
        out = torch.empty(rate_c.shape, dtype=rate_c.dtype, device=rate_c.device)
        for start in range(0, int(rate_c.shape[0]), step):
            out[start:start + step] = uniform_comb_cycle(rate_c[start:start + step], t, self.T, self._offsets)
        return out

    def materialize(self) -> torch.Tensor:
        return torch.stack([uniform_comb_cycle(self._rate_c, t, self.T, self._offsets) for t in range(self.T)], dim=0)

    def is_event_train(self) -> bool:
        return True  # the comb is a 0/1 mask cast to the rate's dtype, every cycle (``to_uniform_spikes``)


class ScaledTrain(CycleTrain):
    """``base * scale``: the eager walk's scaled copy of a train, elementwise per cycle."""

    def __init__(self, base: CycleTrain, scale) -> None:
        self._base, self._scale = base, scale
        super().__init__(base.T)

    def _meta(self) -> tuple:
        base, scale = self._base, self._scale
        if isinstance(scale, torch.Tensor):
            # a dimensioned train times the scale: the eager product's promotion
            return torch.broadcast_shapes(base.shape[1:], scale.shape), torch.result_type(
                torch.empty((1,), dtype=base.dtype), scale), base.device
        return base.shape[1:], base.dtype, base.device

    def cycle(self, t: int) -> torch.Tensor:
        return _as_stacked(self._base[t] * self._scale)

    def materialize(self) -> torch.Tensor:
        return self._base.materialize() * self._scale


class MappedTrain(CycleTrain):
    """``fn`` of the sources' cycles, per cycle (a gather of trains, a train's per-cycle multiplicities); its metadata
    is its first cycle's, read once when asked."""

    def __init__(self, fn: Callable[[list], torch.Tensor], sources: Sequence) -> None:
        self._fn, self._sources = fn, list(sources)
        self._meta_cache: Optional[tuple] = None
        super().__init__(len(self._sources[0]))

    def _meta(self) -> tuple:
        if self._meta_cache is None:
            first = self.cycle(0)
            self._meta_cache = (first.shape, first.dtype, first.device)
        return self._meta_cache

    def cycle(self, t: int) -> torch.Tensor:
        return _as_stacked(self._fn([source[t] for source in self._sources]))


def encode_train(rate: torch.Tensor, T: int, *, phase_dither: bool = False,
                 channels: Optional[torch.Tensor] = None):
    """The uniform re-encode of ``rate``: per cycle where the walk streams, else ``uniform_spike_train``."""
    if streams():
        return EncodedTrain(rate, T, phase_dither=phase_dither, channels=channels)
    return uniform_spike_train(rate, T, phase_dither=phase_dither, channels=channels)


def scale_train(train, scale):
    """``train * scale``, per cycle when ``train`` is read per cycle (a scale broadcast along the cycle axis too is
    applied to the materialized train, as the eager walk applied it)."""
    if not isinstance(train, CycleTrain):
        return train * scale
    if isinstance(scale, torch.Tensor) and scale.dim() > train.dim() - 1:
        return train.materialize() * scale
    return ScaledTrain(train, scale)


def map_cycles(fn: Callable[[list], torch.Tensor], trains: Sequence):
    """``fn`` of every train's cycle t, for every t: per cycle where the walk streams, else stacked (the eager
    gather)."""
    if streams():
        return MappedTrain(fn, trains)
    T = len(trains[0])
    return torch.stack([fn([train[t] for train in trains]) for t in range(T)], dim=0)


def map_train(fn: Callable[[torch.Tensor], torch.Tensor], train):
    """``fn`` of a train, elementwise: per cycle when ``train`` is read per cycle, else on the whole tensor."""
    if isinstance(train, CycleTrain):
        return MappedTrain(lambda cycles: fn(cycles[0]), [train])
    return fn(train)


def train_mean(train) -> torch.Tensor:
    """The mean over the cycles, the eager reduction (a per-cycle train is materialized for it)."""
    return (train.materialize() if isinstance(train, CycleTrain) else train).mean(dim=0)


def is_event_train(train) -> bool:
    """Whether ``train`` is a per-cycle 0/1 event train (the per-cycle law's axon currency)."""
    if train is None:
        return False
    if isinstance(train, CycleTrain):
        return train.is_event_train()
    return bool(torch.all((train == 0) | (train == 1)))


class CycleStack:
    """The cycles of one hop's window, stacked as ``torch.stack`` stacks them. Where the walk streams and the cycles are
    contiguous, each is written into one train allocated at the first cycle, so the cycle list and its stacked copy
    never coexist; otherwise the cycles are listed and stacked (the eager window)."""

    def __init__(self, T: int) -> None:
        self.T = int(T)
        self._cycles: list = []
        self._train: Optional[torch.Tensor] = None
        self._n = 0

    def add(self, cycle: torch.Tensor) -> None:
        if self._n == 0 and streams() and cycle.is_contiguous():
            self._train = cycle.new_empty((self.T, *cycle.shape))
        if self._train is not None:
            if (self._n < self.T and cycle.is_contiguous() and cycle.shape == self._train.shape[1:]
                    and cycle.dtype == self._train.dtype):
                self._train[self._n] = cycle
                self._n += 1
                return
            self._cycles, self._train = [self._train[i].clone() for i in range(self._n)], None
        self._cycles.append(cycle)
        self._n += 1

    def stacked(self) -> torch.Tensor:
        if self._train is not None:
            return self._train if self._n == self.T else self._train[: self._n]
        return torch.stack(self._cycles, dim=0)


class SegmentReaders:
    """How many nodes of a segment still read each node's train: a train no later reader of the segment reads is
    dropped from the walk's tables, except the nodes in ``keep`` (the walk's output)."""

    def __init__(self, seg_nodes, deps_map: dict, keep=()) -> None:
        self._deps = {node: list(deps_map.get(node, [])) for node in seg_nodes}
        self._pending = Counter(dep for deps in self._deps.values() for dep in deps)
        self._keep = set(keep)
        self._read: set = set()

    def _drop(self, node, tables) -> None:
        if self._pending.get(node, 0) <= 0 and node not in self._keep:
            for table in tables:
                table.pop(node, None)

    def read_later(self, node) -> bool:
        """Whether a node of the segment that has not run yet reads ``node``."""
        return self._pending.get(node, 0) > 0

    def read(self, node, *tables: dict) -> None:
        """``node`` has read its sources (once; a repeated call is a no-op): drop each whose readers are all done."""
        if node in self._read:
            return
        self._read.add(node)
        for dep in self._deps[node]:
            self._pending[dep] -= 1
        for dep in self._deps[node]:
            self._drop(dep, tables)

    def unread(self, node, *tables: dict) -> None:
        """``node`` has run: drop its own trains when no node of the segment reads them."""
        self._drop(node, tables)
