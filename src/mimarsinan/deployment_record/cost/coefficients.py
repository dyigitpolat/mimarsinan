"""The cost model's coefficient inventory (schema §7): imported bands + new ones.

The DMA / sync-barrier ENERGY coefficients are **imported** from
``chip_simulation.weight_reuse_cost_model`` — the defensible per-phase model
this program already ships — and applied through ITS functions
(:func:`dma_energy_mj`, :func:`sync_energy_mj`), never copied as numbers.

The coefficients this stage adds are derived from the SANA-FE per-event presets
wherever a per-event basis exists (``chip_simulation/sanafe/presets.py``), and
every one of them ships the written basis inside the :class:`Band` it returns:

* :data:`CORE_INIT` — per-core reset/init energy and time (schema §7 row 5),
* :data:`BYTES_PER_CONNECTIVITY_ENTRY` — the modeled wire width of a span entry,
* :data:`PROGRAMMING_BANDWIDTH_BYTES_PER_S` — programming-payload DMA bandwidth,
* :data:`SYNC_BARRIER_S` — the latency twin of the imported barrier ENERGY band.

``bytes_per_param`` is re-exported for inventory completeness but deliberately
NOT applied by the model: the record's ``params_bytes`` is already a byte count
(``build/payload_sizes.py`` SSOT), so applying a weight width again would
double-count it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from mimarsinan.chip_simulation.sanafe.presets import (
    LOIHI_PRESET,
    TRUENORTH_PRESET,
    PerEventEnergy,
)
from mimarsinan.chip_simulation.weight_reuse_cost_model import (
    DEFAULT_COEFFICIENT_BAND,
    CoefficientBand,
    DmaCostCoefficients,
    phase_cost_model,
)
from mimarsinan.deployment_record.schema import Band

CORNERS: Tuple[str, str, str] = ("low", "nominal", "high")

_OPPOSITE_CORNER: Dict[str, str] = {"low": "high", "nominal": "nominal", "high": "low"}

_J_TO_MJ = 1000.0

# Re-export (never a copy): the model applies THESE coefficient objects.
DMA_COEFFICIENT_BAND: CoefficientBand = DEFAULT_COEFFICIENT_BAND


def require_corner(corner: str) -> str:
    """Fail loud on anything that is not a band corner."""
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {list(CORNERS)}, got {corner!r}")
    return corner


def corner_value(band: Band, corner: str) -> float:
    """The band's value at ``corner``."""
    return float(getattr(band, require_corner(corner)))


def opposite_corner(corner: str) -> str:
    """The corner a DIVIDING coefficient uses so the quotient stays monotone."""
    return _OPPOSITE_CORNER[require_corner(corner)]


def banded(evaluate: Callable[[str], float], *, basis: str) -> Band:
    """A :class:`Band` from a corner-indexed evaluation (``Band`` validates order)."""
    values = [float(evaluate(corner)) for corner in CORNERS]
    return Band(low=values[0], nominal=values[1], high=values[2], basis=basis)


def dma_coefficients(corner: str) -> DmaCostCoefficients:
    """The imported DMA coefficient set at ``corner``."""
    return getattr(DMA_COEFFICIENT_BAND, require_corner(corner))


def _dma_attr_band(attr: str, basis: str) -> Band:
    """A band view of one imported DMA coefficient (values come from the import)."""
    return banded(lambda corner: float(getattr(dma_coefficients(corner), attr)), basis=basis)


E_DMA_PER_BYTE_MJ = _dma_attr_band(
    "e_dma_per_byte_mj",
    "HBM2 ~3.9 pJ/bit / DDR3 ~20 pJ/bit (Horowitz 2014 45 nm) / off-chip worst "
    "case; imported from weight_reuse_cost_model.DEFAULT_COEFFICIENT_BAND",
)
E_SYNC_BARRIER_MJ = _dma_attr_band(
    "e_sync_barrier_mj",
    "~0.1 / ~1 / ~10 uJ per barrier (Loihi-style NoC flush); imported from "
    "weight_reuse_cost_model.DEFAULT_COEFFICIENT_BAND",
)
BYTES_PER_PARAM = _dma_attr_band(
    "bytes_per_param",
    "4- / 8- / 16-bit weights; imported from "
    "weight_reuse_cost_model.DEFAULT_COEFFICIENT_BAND. INVENTORY ONLY: the "
    "record's params_bytes is already a byte count, so the model never applies "
    "this width a second time",
)

BYTES_PER_CONNECTIVITY_ENTRY = Band(
    low=4.0,
    nominal=8.0,
    high=16.0,
    basis=(
        "span entries are exact counts; byte width is modeled until a real chip "
        "wire format exists (chip_spans.txt is a simulator exchange format, not "
        "chip DMA truth). NEW - needs owner sign-off"
    ),
)

PROGRAMMING_BANDWIDTH_BYTES_PER_S = Band(
    low=1.0e9,
    nominal=12.8e9,
    high=256.0e9,
    basis=(
        "programming-payload DMA bandwidth: ~1 GB/s serial configuration port / "
        "DDR3-1600 single channel ~12.8 GB/s / HBM2 stack ~256 GB/s. No chip "
        "programming port is measured anywhere in this program. NEW - needs "
        "owner sign-off"
    ),
)


def _whole_bytes(value: float) -> int:
    """Bytes are counted, never fractional — a fractional payload fails loud."""
    if float(value) != float(int(value)):
        raise ValueError(f"payload bytes must be a whole count, got {value!r}")
    return int(value)


def dma_energy_mj(payload_bytes: float, corner: str) -> float:
    """``e_dma_per_byte`` applied to REAL payload bytes via the existing phase model."""
    return phase_cost_model(
        reprogram_passes=0,
        reuse_passes=0,
        params_reloaded=0,
        activation_bytes_moved=_whole_bytes(payload_bytes),
        coeffs=dma_coefficients(corner),
    ).reuse_dma_mj


def sync_energy_mj(barriers: int, corner: str) -> float:
    """``e_sync_barrier`` × barriers via the existing phase model (sync channel)."""
    return phase_cost_model(
        reprogram_passes=int(barriers),
        reuse_passes=0,
        params_reloaded=0,
        activation_bytes_moved=0,
        coeffs=dma_coefficients(corner),
    ).reprogram_sync_mj


# A core initialization writes every neuron's soma state once; SANA-FE charges
# that as one soma access + one soma update per neuron.  The per-core CONSTANT
# is that per-neuron reset cost over a reference core, so per-core size
# uncertainty lives inside the band instead of a hidden constant.
_REFERENCE_CORE_NEURONS_SMALL = 256   # TrueNorth core (Merolla 2014)
_REFERENCE_CORE_NEURONS_LARGE = 1024  # Loihi core (Davies 2018)

_CORE_INIT_BASIS = (
    "per-core reset/init = (soma_access + soma_update) per neuron x a reference "
    "core, from the SANA-FE per-event presets "
    "(chip_simulation/sanafe/presets.py): TrueNorth over a 256-neuron core "
    "(low), Loihi over 256 (nominal) and 1024 (high) neurons. NEW - needs "
    "owner sign-off"
)


def _reset_energy_j(preset: PerEventEnergy) -> float:
    return preset["soma_access_energy_j"] + preset["soma_update_energy_j"]


def _reset_latency_s(preset: PerEventEnergy) -> float:
    return preset["soma_access_latency_s"] + preset["soma_update_latency_s"]


def _preset_core_band(
    per_neuron: Callable[[PerEventEnergy], float], scale: float, basis: str
) -> Band:
    """The (TrueNorth·small, Loihi·small, Loihi·large) reference-core bracket."""
    corners: Dict[str, float] = {
        "low": per_neuron(TRUENORTH_PRESET) * _REFERENCE_CORE_NEURONS_SMALL,
        "nominal": per_neuron(LOIHI_PRESET) * _REFERENCE_CORE_NEURONS_SMALL,
        "high": per_neuron(LOIHI_PRESET) * _REFERENCE_CORE_NEURONS_LARGE,
    }
    return banded(lambda corner: corners[corner] * scale, basis=basis)


@dataclass(frozen=True)
class CoreInitCoefficients:
    """Per-core reset/init cost (schema §7): banded energy and time, preset-derived."""

    energy_mj: Band
    time_s: Band

    @classmethod
    def from_presets(cls) -> "CoreInitCoefficients":
        """Derive the band from the SANA-FE per-event reset (soma) costs."""
        return cls(
            energy_mj=_preset_core_band(
                _reset_energy_j, _J_TO_MJ, f"{_CORE_INIT_BASIS}; J converted to mJ"
            ),
            time_s=_preset_core_band(_reset_latency_s, 1.0, _CORE_INIT_BASIS),
        )


CORE_INIT = CoreInitCoefficients.from_presets()

# A global barrier is a token traversing the mesh and coming back; the SANA-FE
# presets price one tile hop, so the barrier LATENCY band is that hop cost over
# a mesh-traversal bracket (the ENERGY twin stays the imported coefficient).
_BARRIER_HOPS_SMALL = 16    # 8-hop mesh diameter, out and back
_BARRIER_HOPS_LARGE = 128   # 64-hop mesh diameter, out and back

SYNC_BARRIER_S = banded(
    lambda corner: {
        "low": TRUENORTH_PRESET["tile_hop_latency_s"] * _BARRIER_HOPS_SMALL,
        "nominal": LOIHI_PRESET["tile_hop_latency_s"] * _BARRIER_HOPS_SMALL,
        "high": LOIHI_PRESET["tile_hop_latency_s"] * _BARRIER_HOPS_LARGE,
    }[corner],
    basis=(
        "barrier latency = SANA-FE tile_hop_latency_s (TrueNorth 4 ns / Loihi "
        "5 ns) x a 16-hop (8-hop diameter, out and back) to 128-hop mesh "
        "traversal. NEW - needs owner sign-off"
    ),
)

__all__ = [
    "BYTES_PER_CONNECTIVITY_ENTRY",
    "BYTES_PER_PARAM",
    "CORE_INIT",
    "CORNERS",
    "CoreInitCoefficients",
    "DMA_COEFFICIENT_BAND",
    "E_DMA_PER_BYTE_MJ",
    "E_SYNC_BARRIER_MJ",
    "PROGRAMMING_BANDWIDTH_BYTES_PER_S",
    "SYNC_BARRIER_S",
    "banded",
    "corner_value",
    "dma_coefficients",
    "dma_energy_mj",
    "opposite_corner",
    "require_corner",
    "sync_energy_mj",
]
