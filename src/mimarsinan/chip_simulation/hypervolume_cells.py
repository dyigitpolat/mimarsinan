"""The full-tuple hypervolume coverage cell and claimed sub-product enumeration."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.hypervolume_axes import (
    AXES,
    AXIS_WILDCARD,
    HypervolumeAxis,
    collapse_orthogonal_axes,
    collapsed_axis_representatives,
)

_CELL_AXES: Tuple[str, ...] = tuple(a.name for a in collapse_orthogonal_axes(AXES))

_COLLAPSED_REPRESENTATIVES: Dict[str, str] = collapsed_axis_representatives(AXES)


@dataclass(frozen=True)
class HypervolumeCell:
    """The FULL-TUPLE coverage cell — extends the (firing × sync × backend) ``CertificationCell`` to the config tuple.

    ``depth`` is a real coordinate so a shallow INVALID and a deeper VALID_FLAGGED at the
    same recipe do not collapse; the collapsed ``encoding_placement`` axis is NOT a field.
    """

    firing: str
    sync: Optional[str]
    backend: str
    vehicle: str
    dataset: str
    regime: str
    quantization: str
    pruning: str
    mapping_strategy: str
    s: str
    depth: str

    @property
    def cert_cell(self) -> CertificationCell:
        """The embedded (firing × sync × backend) E6 cell, folding a collapsed backend to its representative."""
        sync = None if self.sync in (None, "none", "") else self.sync
        backend = _COLLAPSED_REPRESENTATIVES.get("backend", self.backend)
        return CertificationCell(firing=self.firing, sync=sync, backend=backend)

    @property
    def cell_key(self) -> str:
        """Canonical full-tuple key: the ``mode[/schedule]@backend`` cert prefix + ``axis=value`` segments."""
        parts = [self.cert_cell.cell_key]
        for axis in _CELL_AXES:
            if axis in ("firing", "sync", "backend"):
                continue
            parts.append(f"{axis}={getattr(self, _FIELD_FOR_AXIS[axis])}")
        return "|".join(parts)

    @classmethod
    def from_key(cls, key: str) -> "HypervolumeCell":
        """Parse a canonical full-tuple key back into a cell (dropped collapsed axes default to their representative)."""
        cert_part, *rest = key.split("|")
        cert = CertificationCell.from_key(cert_part)
        kv = {}
        for seg in rest:
            if "=" not in seg:
                raise ValueError(f"malformed hypervolume cell segment {seg!r} in {key!r}")
            axis, value = seg.split("=", 1)
            kv[axis] = value

        def _ext(axis: str) -> str:
            if axis in kv:
                return kv[axis]
            rep = _COLLAPSED_REPRESENTATIVES.get(axis)
            if rep is not None:
                return rep
            raise KeyError(axis)

        return cls(
            firing=cert.firing,
            sync=cert.sync,
            backend=cert.backend,
            vehicle=_ext("vehicle"),
            dataset=_ext("dataset"),
            regime=_ext("regime"),
            quantization=_ext("quantization"),
            pruning=_ext("pruning"),
            mapping_strategy=_ext("mapping_strategy"),
            s=_ext("S"),
            depth=_ext("depth"),
        )


_FIELD_FOR_AXIS: Dict[str, str] = {a: a for a in _CELL_AXES}
_FIELD_FOR_AXIS["S"] = "s"

_MATCH_FIELDS: Tuple[str, ...] = tuple(_FIELD_FOR_AXIS[a] for a in _CELL_AXES)


def _coord(cell: "HypervolumeCell", field_name: str) -> str:
    """A cell's coordinate value for a match field, with ``sync=None`` read as ``none``."""
    value = getattr(cell, field_name)
    if field_name == "sync" and value in (None, ""):
        return "none"
    return str(value)


def cell_covers(claimed: "HypervolumeCell", covered: "HypervolumeCell") -> bool:
    """True iff a COVERED cell satisfies a CLAIMED cell — a claimed :data:`AXIS_WILDCARD` matches any covered value, else exact."""
    for field_name in _MATCH_FIELDS:
        claim_value = _coord(claimed, field_name)
        if claim_value == AXIS_WILDCARD:
            continue
        if claim_value != _coord(covered, field_name):
            return False
    return True


_CLAIMED_DEFAULTS: Dict[str, Tuple[str, ...]] = {
    "firing": ("ttfs_cycle_based",),
    "sync": ("cascaded",),
    "backend": ("sanafe",),
    "vehicle": ("deep_cnn",),
    "dataset": ("mnist",),
    "regime": ("from_scratch",),
    "quantization": ("none",),
    "pruning": ("dense",),
    "mapping_strategy": ("packed",),
    "S": (AXIS_WILDCARD,),
    "depth": (AXIS_WILDCARD,),
}


_WILDCARD_DEFAULT_AXES: Tuple[str, ...] = ("S", "depth")


def _enumerate_claim(chosen: Mapping[str, Sequence[str]]) -> List[HypervolumeCell]:
    """Build the deduped cartesian product of per-axis value lists into cells (collapsed axes fold to one)."""
    ordered_axes = list(_CELL_AXES)

    def field(kv: Dict[str, str], axis: str) -> str:
        if axis in kv:
            return kv[axis]
        return _COLLAPSED_REPRESENTATIVES.get(axis, "")

    cells: List[HypervolumeCell] = []
    for combo in itertools.product(*(tuple(chosen[a]) for a in ordered_axes)):
        kv = dict(zip(ordered_axes, combo))
        raw_sync = field(kv, "sync")
        sync = None if raw_sync in (None, "none", "") else raw_sync
        cells.append(
            HypervolumeCell(
                firing=field(kv, "firing"),
                sync=sync if sync is not None else "none",
                backend=field(kv, "backend"),
                vehicle=field(kv, "vehicle"),
                dataset=field(kv, "dataset"),
                regime=field(kv, "regime"),
                quantization=field(kv, "quantization"),
                pruning=field(kv, "pruning"),
                mapping_strategy=field(kv, "mapping_strategy"),
                s=field(kv, "S"),
                depth=field(kv, "depth"),
            )
        )
    seen: Dict[str, HypervolumeCell] = {}
    for cell in cells:
        seen.setdefault(cell.cell_key, cell)
    return list(seen.values())


def claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """Enumerate a claimed sub-product as concrete cells; unpinned axes take their single screened default.

    The LEGACY single-default claim — for the honest enumerated denominator use
    :func:`honest_claimed_subproduct`.
    """
    chosen: Dict[str, Tuple[str, ...]] = {}
    for axis in _CELL_AXES:
        if axis in axis_values:
            chosen[axis] = tuple(axis_values[axis])
        else:
            chosen[axis] = _CLAIMED_DEFAULTS[axis]
    return _enumerate_claim(chosen)


def honest_claimed_subproduct(**axis_values: Sequence[str]) -> List[HypervolumeCell]:
    """The HONEST claimed sub-product whose denominator CONSUMES each unpinned axis's screening status.

    A SCREENED_COLLAPSED axis contributes one representative; an ENUMERATED_INTERACTING /
    ASSERTED_UNSCREENED axis is enumerated over its full domain (S / depth stay wildcard).
    """
    chosen: Dict[str, Tuple[str, ...]] = {}
    for axis in _CELL_AXES:
        if axis in axis_values:
            chosen[axis] = tuple(axis_values[axis])
            continue
        if axis in _WILDCARD_DEFAULT_AXES:
            chosen[axis] = _CLAIMED_DEFAULTS[axis]
            continue
        chosen[axis] = HypervolumeAxis.get(axis).values
    return _enumerate_claim(chosen)
