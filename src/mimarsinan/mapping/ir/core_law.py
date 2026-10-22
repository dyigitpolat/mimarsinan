"""``CoreLaw``: the per-column integer readout law a neural core carries (E282.S)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Iterable, List, Sequence

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal

_FLOAT_FIELDS = frozenset({"input_step", "output_step"})


@dataclass(frozen=True, eq=False)
class CoreLaw:
    """Per output column: the DAC step of the column's inputs, f_adc (t, L), and the requantization
    ``y = clamp(rnd_s(M0 f_adc(acc) + B), lo, hi)`` onto the output grid of step ``output_step``.

    Shared, never copied per core: bank-backed position cores reference one law per bank slice, and
    :meth:`take_columns` returns one object per distinct column selection (pickle stores it once).
    """

    input_step: np.ndarray
    m0: np.ndarray
    shift: np.ndarray
    bias: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    output_step: np.ndarray
    adc_shift: np.ndarray
    adc_levels: np.ndarray

    def __post_init__(self) -> None:
        width = None
        for f in fields(self):
            dtype = np.float64 if f.name in _FLOAT_FIELDS else np.int64
            array = np.array(getattr(self, f.name), dtype=dtype).reshape(-1)
            array.flags.writeable = False
            if width is None:
                width = array.size
            if array.size != width:
                raise ValueError(f"CoreLaw: field {f.name!r} has {array.size} columns, expected {width}")
            object.__setattr__(self, f.name, array)
        object.__setattr__(self, "_memo", {})

    @property
    def columns(self) -> int:
        return int(self.m0.size)

    def take_columns(self, keep: Sequence[int]) -> "CoreLaw":
        """The law of the kept columns, in ``keep`` order; one object per distinct ``keep`` content."""
        index = np.asarray(keep, dtype=np.intp).reshape(-1)
        key = index.tobytes()
        memo = self._memo  # pyright: ignore[reportAttributeAccessIssue] -- set in __post_init__
        hit = memo.get(key)
        if hit is None:
            hit = CoreLaw(**{f.name: getattr(self, f.name)[index] for f in fields(self)})
            memo[key] = hit
        return hit

    @classmethod
    def concat(cls, laws: Iterable["CoreLaw"]) -> "CoreLaw":
        laws = list(laws)
        return cls(**{f.name: np.concatenate([getattr(law, f.name) for law in laws]) for f in fields(cls)})

    @classmethod
    def blank(cls, columns: int) -> "CoreLaw":
        """Unused columns: every output is code 0."""
        zeros = np.zeros(int(columns))
        return cls(**{f.name: zeros for f in fields(cls)})

    def splice(self, offset: int, law: "CoreLaw") -> "CoreLaw":
        """This law with columns ``[offset, offset + law.columns)`` replaced by ``law``'s."""
        spliced = {}
        for f in fields(self):
            array = np.array(getattr(self, f.name))
            array[int(offset): int(offset) + law.columns] = getattr(law, f.name)
            spliced[f.name] = array
        return CoreLaw(**spliced)

    def column_digests(self) -> List[str]:
        """One stable digest per column (floats by their exact hex), for digests over surviving columns."""
        out = []
        for j in range(self.columns):
            parts = []
            for f in fields(self):
                value = getattr(self, f.name)[j]
                parts.append(f"{f.name}={float(value).hex() if f.name in _FLOAT_FIELDS else int(value)}")
            out.append(hashlib.sha256(";".join(parts).encode()).hexdigest())
        return out

    def digest(self) -> str:
        return hashlib.sha256("|".join(self.column_digests()).encode()).hexdigest()

    def __reduce__(self):
        """One contiguous buffer (every field's 8-byte values, floats by their bits) and the width, so a
        pickle stores each distinct law as a few memo entries, never one numpy array per field."""
        rows = [np.ascontiguousarray(getattr(self, f.name)).view(np.int64) for f in fields(self)]
        return _core_law_from_buffer, (np.stack(rows).astype("<i8").tobytes(), self.columns)


def _core_law_from_buffer(buffer: bytes, width: int) -> CoreLaw:
    """The inverse of :meth:`CoreLaw.__reduce__`; the rebuilt law has a fresh ``take_columns`` memo."""
    names = [f.name for f in fields(CoreLaw)]
    matrix = np.frombuffer(buffer, dtype="<i8").reshape(len(names), int(width)).astype(np.int64)
    return CoreLaw(**{
        name: matrix[k].view(np.float64) if name in _FLOAT_FIELDS else matrix[k] for k, name in enumerate(names)
    })


def graph_carries_laws(ir_graph) -> bool:
    """Whether any core of ``ir_graph`` carries a readout law (the integer value path)."""
    return any(getattr(node, "readout_law", None) is not None for node in ir_graph.nodes)


class ReadoutLawMixedCoreError(ToolchainRefusal, ValueError):
    """A hard core asked to hold soft cores with and without a readout law: one readout per hard core."""


def merge_core_law(hard_core, softcore, *, axon_offset: int, neuron_offset: int) -> None:
    """Merge ``softcore``'s law into ``hard_core`` over the neuron range it occupies (``hardware_bias``'s rule).

    A soft core filling the whole hard core shares its law object; nothing is set when neither carries one.
    """
    law = getattr(softcore, "readout_law", None)
    current = getattr(hard_core, "readout_law", None)
    placed_before = axon_offset > 0 or neuron_offset > 0
    if (law is None) != (current is None) and (law is None or placed_before):
        raise ReadoutLawMixedCoreError(
            f"soft core {getattr(softcore, 'id', '?')} {'has no' if law is None else 'has a'} readout law but the "
            f"hard core already holds soft cores that {'do' if law is None else 'do not'}"
        )
    if law is None:
        return
    if current is None and neuron_offset == 0 and law.columns == int(hard_core.neurons_per_core):
        hard_core.readout_law = law
        return
    base = current if current is not None else CoreLaw.blank(int(hard_core.neurons_per_core))
    hard_core.readout_law = base.splice(neuron_offset, law)
