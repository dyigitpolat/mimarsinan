"""Resolve platform core lists into IR-mapping parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from mimarsinan.common.refusal import ToolchainRefusal

#: A core type's weight-sign granularity: one sign per synapse (absent), or one static sign per axon row for all
#: the row's neurons (ODIN's ``SPI_SYN_SIGN``), where a logical input with mixed-sign weights takes two rows.
PER_SYNAPSE = "per_synapse"
PER_AXON = "per_axon"
WEIGHT_SIGN_GRANULARITIES = (PER_SYNAPSE, PER_AXON)
SIGN_GRANULARITY_FIELD = "weight_sign_granularity"


class SignGranularityError(ToolchainRefusal, ValueError):
    """A weight-sign granularity declaration the mapper cannot plan, refused by name."""


def pool_sign_granularity(cores: Sequence[Mapping[str, Any]]) -> str:
    """The pool's granularity: per_axon if any core type declares it (rows that each carry one sign are valid on
    every core, the conservative fold ``has_bias`` takes too), else per_synapse."""
    declared = {core.get(SIGN_GRANULARITY_FIELD, PER_SYNAPSE) for core in cores}
    unknown = sorted(str(v) for v in declared - set(WEIGHT_SIGN_GRANULARITIES))
    if unknown:
        raise SignGranularityError(
            f"{SIGN_GRANULARITY_FIELD} {unknown} is not one of {list(WEIGHT_SIGN_GRANULARITIES)}")
    return PER_AXON if PER_AXON in declared else PER_SYNAPSE


def carries_sign_register(axons: int, neurons: int, cores: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a packed core of this geometry holds a per-row sign register: a core of a declared per-axon type, or a
    fused core (no declared type of its geometry) with a per-axon component, the fold a fused core takes
    (``hard_core_mapping``). A shape-only plane sees geometry, not the declared type the packer takes, so where two
    declared types of opposite granularity share a geometry it counts the register; the record signs per core."""
    exact = [c for c in cores if (int(c["max_axons"]), int(c["max_neurons"])) == (int(axons), int(neurons))]
    parts = exact or [c for c in cores
                      if int(c["max_neurons"]) == int(neurons) and int(axons) % int(c["max_axons"]) == 0]
    return any(c.get(SIGN_GRANULARITY_FIELD, PER_SYNAPSE) == PER_AXON for c in parts)


def operand_sign_conflict(cores: Sequence[Any]) -> Optional[str]:
    """Why per-axon weight sign and an operand core type (N2) cannot share a chip on this tree, else None: the
    per-axon split fixes each physical row's sign once, from the static weights (``softcore/sign_split.py``), and an
    operand bank's payload is written per inference, so its signs move with every sample. The composition (the split
    and ``rows_per_input`` decided per core role, on every plane that reads the pool's granularity) is not built."""
    axon = [i for i, c in enumerate(cores) if isinstance(c, Mapping) and c.get(SIGN_GRANULARITY_FIELD) == PER_AXON]
    operand = [i for i, c in enumerate(cores) if isinstance(c, Mapping) and c.get("operand") is not None]
    if not axon or not operand:
        return None
    return (f"per-axon weight sign and an operand core type do not share a chip on this tree: cores {axon} declare "
            f"{SIGN_GRANULARITY_FIELD} '{PER_AXON}' and cores {operand} declare operand; the per-axon split fixes "
            "each row's sign once from static weights, and an operand bank's payload is written per inference "
            f"(declare '{PER_SYNAPSE}' on every core type, or no operand core type)")


def rows_per_input(granularity: str) -> int:
    """Physical rows one logical input may take: two under per-axon sign (an excitatory and an inhibitory row)."""
    return 2 if granularity == PER_AXON else 1


@dataclass(frozen=True)
class PlatformMappingParams:
    hardware_bias: bool
    effective_max_axons: int
    effective_max_neurons: int
    allow_coalescing: bool
    rows_per_input: int = 1

    @property
    def row_capacity(self) -> int:
        """Rows the capacity bound divides by: the tiler's limit under per-synapse sign (as before), and under
        per-axon sign the physical rows a full tile of mixed rows takes (its inputs and bias row, two rows each)."""
        if self.rows_per_input == 1:
            return self.effective_max_axons
        return self.rows_per_input * (self.effective_max_axons + (0 if self.hardware_bias else 1))


def resolve_scalar_mapping_params(
    *,
    max_axons: int,
    max_neurons: int,
    hardware_bias: bool = False,
    allow_coalescing: bool = False,
) -> PlatformMappingParams:
    """Resolve tiling limits when only scalar max_axons/max_neurons are known."""
    effective_max_axons = int(max_axons) if hardware_bias else int(max_axons) - 1
    return PlatformMappingParams(
        hardware_bias=bool(hardware_bias),
        effective_max_axons=effective_max_axons,
        effective_max_neurons=int(max_neurons),
        allow_coalescing=bool(allow_coalescing),
    )


def resolve_platform_mapping_params(
    cores: Sequence[dict[str, Any]],
    *,
    allow_coalescing: bool = False,
) -> PlatformMappingParams:
    """The tiler's per-core limits: ``floor(max_axons / r)`` logical inputs (``r`` rows per input under the pool's
    sign granularity), less the bias row when the cores carry no hardware bias."""
    if not cores:
        raise ValueError("cores must be a non-empty list")
    hardware_bias = all(bool(ct.get("has_bias", True)) for ct in cores)
    max_axons = max(int(ct["max_axons"]) for ct in cores)
    max_neurons = max(int(ct["max_neurons"]) for ct in cores)
    rows = rows_per_input(pool_sign_granularity(cores))
    effective_max_axons = max_axons // rows - (0 if hardware_bias else 1)
    if rows > 1 and effective_max_axons < 1:
        raise SignGranularityError(
            f"per-axon cores of max_axons {max_axons} hold no logical input beside the bias row: each input "
            f"takes {rows} rows")
    return PlatformMappingParams(
        hardware_bias=hardware_bias,
        effective_max_axons=effective_max_axons,
        effective_max_neurons=max_neurons,
        allow_coalescing=bool(allow_coalescing),
        rows_per_input=rows,
    )
