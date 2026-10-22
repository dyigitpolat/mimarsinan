"""[E190] The validity domain of aggregate constants.

A published aggregate (a per-core footprint, a whole-system energy per event) holds at the core it
was measured on: geometry, cell bits and chip core count. Outside it the aggregate refuses by name,
unless the profile declares a scaling rule; decomposed constants price wherever they are declared.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from mimarsinan.deployment_record.cost.absolute.context import PricingRefusal
from mimarsinan.deployment_record.platform_physics.constants import keys_in_group
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics

AGGREGATE_CONSTANTS: Tuple[str, ...] = keys_in_group("aggregate")


@dataclass(frozen=True)
class CoreGeometry:
    """One declared core type of the priced chip."""

    axons: int
    neurons: int
    count: int


def geometry_of(cores: Sequence[Mapping[str, Any]]) -> Tuple[CoreGeometry, ...]:
    """The declared core types (a missing ``count`` is one core, the platform convention)."""
    return tuple(CoreGeometry(int(ct["max_axons"]), int(ct["max_neurons"]), int(ct.get("count", 1)))
                 for ct in cores)


def declared_geometry(cores: Sequence[Mapping[str, Any]]) -> Optional[Tuple[CoreGeometry, ...]]:
    """The declared core types, or None when the declaration states no complete core geometry."""
    if not cores or any("max_axons" not in ct or "max_neurons" not in ct for ct in cores):
        return None
    return geometry_of(cores)


class Unjudged:
    """A caller's statement that its census states no core geometry (``UNJUDGED``): the aggregates
    are read as declared, by name at the call site rather than by omission."""

    def __repr__(self) -> str:
        return "UNJUDGED"


UNJUDGED = Unjudged()


class CensusProved:
    """A caller's statement that only what its census proves may be judged (``CENSUS_PROVED``): a verdict
    resting on the census's mean core refuses by name. For surfaces that report values without their
    sources (the cross-platform table), where a stated mean-core basis would not reach the reader."""

    def __repr__(self) -> str:
        return "CENSUS_PROVED"


CENSUS_PROVED = CensusProved()


@dataclass(frozen=True)
class CensusCells:
    """What a census proves without the declared core types: its core count and, where stated, its cell
    count, never one core type of several cores (types that share a dimension average to any whole mean
    core: 384 x 256 + 128 x 256 reads as two 256 x 256 cores). A declared 'cells' scaling rule judges it
    exactly by its cells; at a point validity a whole mean core off the point refuses (one core lies off
    it), and one at the point is judged there with that basis stated in the term (``mean_core_note``),
    or refused by name under ``CENSUS_PROVED``."""

    cores: int
    cells: Optional[float]
    #: The whole mean core (axons, neurons) where the cells identity holds, else None.
    mean: Optional[Tuple[int, int]]
    reason: str
    #: Set under ``CENSUS_PROVED``: a mean-core verdict refuses by name.
    strict: bool = False


def _whole(value: float) -> Any:
    return int(value) if value == int(value) else value


def census_geometry(quantities: Any) -> Optional[Union[Tuple[CoreGeometry, ...], CensusCells]]:
    """The census's own core type where it proves one (a single core), else what it proves of the chip
    (``CensusCells``); a census that states no core geometry gives None (read as declared)."""
    keys = ("cores_physical", "axons_physical", "neurons_physical")
    if not all(quantities.has(key) for key in keys):
        return None
    cores, axons, neurons = (quantities.get(key).value for key in keys)
    if cores <= 0:
        return None
    cells = quantities.get("cells_physical").value if quantities.has("cells_physical") else None
    if cores != int(cores) or axons % cores or neurons % cores:
        return CensusCells(int(cores), cells, None, (
            f"it pools several core types: axons_physical {_whole(axons)} and neurons_physical "
            f"{_whole(neurons)} are not whole per core over {_whole(cores)} cores"))
    n, a, b = int(cores), int(axons // cores), int(neurons // cores)
    if cells is None:
        return CensusCells(n, None, None, "it states no cells_physical, so its cells cannot show one core type")
    if cells != n * a * b:
        return CensusCells(n, cells, None, (f"it pools several core types: cells_physical {_whole(cells)} is not "
                                            f"cores x (axons / cores) x (neurons / cores) = {n * a * b}"))
    if n == 1:
        return (CoreGeometry(a, b, 1),)
    return CensusCells(n, cells, (a, b), (
        f"it proves only its cell count: {n} cores averaging {a} x {b} also arise from core types that "
        "share a dimension"))


def judged_geometry(geometry: Any, quantities: Any) -> Optional[Union[Tuple[CoreGeometry, ...], CensusCells]]:
    """The core types the aggregates are judged at: the caller's, else what the census proves (strictly
    under ``CENSUS_PROVED``), and none only where the caller says ``UNJUDGED`` or the census states no
    core geometry."""
    if isinstance(geometry, Unjudged):
        return None
    if geometry is None or isinstance(geometry, CensusProved):
        proved = census_geometry(quantities)
        if isinstance(geometry, CensusProved) and isinstance(proved, CensusCells):
            return replace(proved, strict=True)
        return proved
    return tuple(geometry)


def _census_scale(point: Any, chip: CensusCells, where: str) -> Tuple[Optional[Tuple[float, ...]], Optional[str]]:
    """A census judged at a point: through a declared 'cells' rule by its cell count, else refused."""
    if point.scaling == "cells" and chip.cells is not None:
        return (chip.cells / (chip.cores * point.axons * point.neurons),), None
    if chip.mean is not None and chip.mean != (point.axons, point.neurons):
        return None, (f"{where}; this chip's {chip.cores} cores average {chip.mean[0]} x {chip.mean[1]}, so at "
                      "least one lies off it, and the profile declares no scaling rule")
    if chip.mean is None or chip.strict:
        return None, (f"{where}; the census does not prove one core type ({chip.reason}), so it names no core "
                      "to judge: pass the declared core types")
    return (1.0,), None


def mean_core_note(physics: PlatformPhysics, chip: Any, *, point_model: bool = False) -> str:
    """The basis an aggregate judged at a census's mean core states (``_census_scale``), else ''. A
    'cells' rule reads a footprint exactly by the census's cells; a ``point_model`` (a per-event
    aggregate) is judged at the mean core under any rule."""
    point = physics.validity.geometry
    if not isinstance(chip, CensusCells) or chip.mean is None or point is None:
        return ""
    if not point_model and point.scaling == "cells" and chip.cells is not None:
        return ""
    return (f"; judged at the census's mean core: {chip.cores} cores averaging {chip.mean[0]} x {chip.mean[1]} "
            "prove only their cell count (pass the declared core types to judge each type)")


def footprint_cores(chip: Union[Sequence[CoreGeometry], CensusCells], scales: Sequence[float]) -> float:
    """The point-core footprints a chip occupies at its per-type scales (``aggregate_scales``)."""
    if isinstance(chip, CensusCells):
        return chip.cores * scales[0]
    return sum(core.count * scale for core, scale in zip(chip, scales))


def aggregate_scales(physics: PlatformPhysics, chip: Union[Sequence[CoreGeometry], CensusCells],
                     weight_bits: Optional[int]) -> Tuple[Optional[Tuple[float, ...]], Optional[str]]:
    """Per core type, the factor an aggregate is read at (1 inside its validity), or the reason it
    refuses. Physics without a structured validity (an archived block, an operator's custom one)
    states none and is read as declared."""
    point = physics.validity.geometry
    census = chip if isinstance(chip, CensusCells) else None
    cores: Sequence[CoreGeometry] = () if isinstance(chip, CensusCells) else chip
    if point is None:
        return tuple(1.0 for _ in cores), None
    where = f"{physics.name}'s aggregates hold at its {point.axons} x {point.neurons} core"
    if point.cell_bits is not None and weight_bits is not None and int(weight_bits) > point.cell_bits:
        return None, (f"{where} of {point.cell_bits}-bit cells; this program's weights take "
                      f"{weight_bits} bits")
    total = census.cores if census is not None else sum(core.count for core in cores)
    if point.core_count is not None and not point.core_count[0] <= total <= point.core_count[1]:
        return None, (f"{where} on chips of {point.core_count[0]}-{point.core_count[1]} cores; this "
                      f"chip declares {total}")
    if census is not None:
        return _census_scale(point, census, where)
    scales = []
    for core in cores:
        if (core.axons, core.neurons) == (point.axons, point.neurons):
            scales.append(1.0)
        elif point.scaling == "cells":
            scales.append(core.axons * core.neurons / (point.axons * point.neurons))
        else:
            return None, (f"{where}; this chip declares a {core.axons} x {core.neurons} core and the "
                          "profile declares no scaling rule")
    return tuple(scales), None


def validity_refusals(physics: PlatformPhysics, cores: Sequence[Mapping[str, Any]], *,
                      weight_bits: Optional[int] = None) -> Tuple[PricingRefusal, ...]:
    """Every aggregate constant the profile declares that this chip lies outside of, by name."""
    _scales, reason = aggregate_scales(physics, geometry_of(cores), weight_bits)
    if reason is None:
        return ()
    return tuple(PricingRefusal(name=key, reason=reason) for key in AGGREGATE_CONSTANTS if physics.has(key))
