"""The XCU55C device totals, the availability scenarios, and the packing bound.

Every number here is DATASHEET provenance -- a published product-table figure or
an arithmetic reduction of one -- and carries the sentence that says so. The one
thing this module does NOT know is how much of the device the Alveo shell keeps:
that is a `report_utilization` on the linked design, which exists only on HACC,
so it appears here as a NAMED ASSUMPTION and never as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

#: The device the Alveo U55C card carries (AMD DS978, "FPGA Resource Information").
DEVICE_PART = "XCU55C"
DEVICE_CARD = "AMD Alveo U55C"

#: The resource classes the packing bound is taken over, in the order the study
#: reports them. `bram36` is counted in 36 Kb tiles, which is how the device is
#: specified; `lut_sites` is the LUT-equivalent cell count PLUS the LUT6 sites a
#: distributed RAM occupies, because both compete for the same CLB resource.
RESOURCE_CLASSES: Tuple[str, ...] = ("lut_sites", "flip_flops", "bram36", "uram")

#: What each resource class is called on the product table, so a bound can be read.
CENSUS_TO_DEVICE: Mapping[str, str] = {
    "lut_sites": "luts",
    "flip_flops": "registers",
    "bram36": "bram36_tiles",
    "uram": "uram_blocks",
}

#: One `RAM64M8` is eight 64x1 LUT RAMs in one SLICEM: 512 bits of storage held
#: in 8 LUT6 sites (AMD UG574, UltraScale architecture CLB). A distributed-RAM
#: cell this table does not name is a REFUSAL, never a rounding.
LUTRAM_LUT_SITES: Mapping[str, int] = {"RAM64M8": 8}
LUTRAM_BITS: Mapping[str, int] = {"RAM64M8": 8 * 64}
LUTRAM_PROVENANCE = (
    "one `RAM64M8` is eight 64x1 LUT RAMs in one SLICEM, i.e. 512 bits held in "
    "8 LUT6 sites (AMD UG574, UltraScale architecture CLB)")


def _lutram_total(cells: Mapping[str, int], table: Mapping[str, int],
                  *, quantity: str) -> int:
    unknown = sorted(set(cells) - set(table))
    if unknown:
        raise ValueError(
            f"no {quantity} conversion for the distributed-RAM cells {unknown}; "
            f"add them with their UG574 provenance rather than letting the "
            f"study round them away")
    return sum(table[name] * int(count) for name, count in cells.items())


def lutram_lut_sites(census: Mapping[str, Any]) -> int:
    """The LUT6 sites this census's distributed RAM occupies."""
    return _lutram_total(
        census.get("lutram_cells", {}), LUTRAM_LUT_SITES, quantity="LUT-site")


def lutram_bits(census: Mapping[str, Any]) -> int:
    """The storage this census's distributed RAM holds, in bits."""
    return _lutram_total(
        census.get("lutram_cells", {}), LUTRAM_BITS, quantity="bit-capacity")


def census_costs(census: Mapping[str, Any]) -> Dict[str, float]:
    """One census as the resource classes the bound is taken over."""
    return {
        "lut_sites": float(census["lut_equivalent"]) + lutram_lut_sites(census),
        "flip_flops": float(census["flip_flops"]),
        "bram36": float(census["bram36"]) + float(census["bram18"]) / 2.0,
        "uram": float(census["uram"]),
    }

#: The published figures. `published` is the string the product table prints;
#: `value` is the number the arithmetic uses; `derivation` says how one becomes
#: the other. NOTHING here is measured on a board.
DEVICE_TOTALS: Mapping[str, Dict[str, Any]] = {
    "luts": {
        "value": 1_304_000, "published": "1,304K",
        "derivation": "product-table LUT count, taken as published",
    },
    "registers": {
        "value": 2_607_000, "published": "2,607K",
        "derivation": "product-table register count, taken as published",
    },
    "bram36_tiles": {
        "value": 2_016, "published": "70.9 Mb total block RAM",
        "derivation": "70.9 Mib / 36 Kib per RAMB36 tile = 2,016 tiles",
    },
    "uram_blocks": {
        "value": 960, "published": "270 Mb UltraRAM",
        "derivation": "270 Mib / 288 Kib per URAM288 block = 960 blocks",
    },
    "dsp_slices": {
        "value": 9_024, "published": "9,024",
        "derivation": "product-table DSP count; no configuration in this study "
                      "uses a DSP, so it never binds",
    },
}

DEVICE_PROVENANCE = (
    "datasheet -- AMD Alveo U55C product table / DS978 (XCU55C, three SLRs, "
    "16 GB HBM2). No figure in this table was measured on hardware."
)

#: The ONE thing the datasheet cannot say. P7b's `report_utilization` on the
#: linked design is the only truth about it.
SHELL_OVERHEAD_STATUS = (
    "UNKNOWN UNTIL P7b: the Alveo shell (XDMA/HBM AXI infrastructure, the "
    "dynamic-region boundary) consumes device resources this study has not "
    "measured. A Vivado `report_utilization` on the linked "
    "`xilinx_u55c_gen3x16_xdma_base_3` design is the only source of that "
    "number, and Vivado exists only on HACC (owner-gated login), so it is the "
    "B0-adjacent step of the BOARD half of P8."
)

#: The assumed shell reservation of the conservative scenario. An ASSUMPTION,
#: stated so it can be replaced by one measurement rather than argued about.
ASSUMED_SHELL_FRACTION = 0.30


@dataclass(frozen=True)
class Availability:
    """One answer to `how much of the device is ours`, with its status."""

    key: str
    label: str
    shell_fraction: float
    status: str

    def available(self) -> Dict[str, int]:
        """Device totals less the scenario's shell reservation, per class."""
        return {
            column: int(DEVICE_TOTALS[column]["value"] * (1.0 - self.shell_fraction))
            for column in CENSUS_TO_DEVICE.values()
        }

    def as_record(self) -> Dict[str, Any]:
        return {
            "key": self.key, "label": self.label,
            "shell_fraction": self.shell_fraction, "status": self.status,
            "available": self.available(),
        }


AVAILABILITY_SCENARIOS: Tuple[Availability, ...] = (
    Availability(
        key="datasheet_total",
        label="datasheet total (naive: the whole device is the user's)",
        shell_fraction=0.0,
        status="datasheet, but NOT achievable -- it ignores the shell entirely",
    ),
    Availability(
        key="assumed_shell_30pct",
        label=f"conservative: the shell is ASSUMED to reserve "
              f"{ASSUMED_SHELL_FRACTION:.0%} of every class",
        shell_fraction=ASSUMED_SHELL_FRACTION,
        status=f"ASSUMPTION, not a measurement. {SHELL_OVERHEAD_STATUS}",
    ),
)


@dataclass(frozen=True)
class ResourceBound:
    """One resource class's answer to `how many cores fit`."""

    resource: str
    per_core: float
    fixed_overhead: float
    available: int
    n_max: int | None

    def as_record(self) -> Dict[str, Any]:
        return {
            "resource": self.resource, "per_core": self.per_core,
            "fixed_overhead": self.fixed_overhead, "available": self.available,
            "n_max": self.n_max,
        }


def packing_bound(
    *, per_core: Mapping[str, float], fixed_overhead: Mapping[str, float],
    availability: Availability,
) -> Tuple[ResourceBound, ...]:
    """`N_max(r) = floor((available(r) - fixed(r)) / per_core(r))`, per class.

    A class a configuration does not use at all (per_core 0) has NO bound rather
    than an infinite one: reporting infinity as a number is how a non-binding
    class ends up quoted as the headline.
    """
    budget = availability.available()
    bounds = []
    for resource in RESOURCE_CLASSES:
        column = CENSUS_TO_DEVICE[resource]
        cost = float(per_core.get(resource, 0.0))
        fixed = float(fixed_overhead.get(resource, 0.0))
        room = budget[column] - fixed
        bounds.append(ResourceBound(
            resource=resource, per_core=cost, fixed_overhead=fixed,
            available=budget[column],
            n_max=None if cost <= 0 else max(0, int(room // cost)),
        ))
    return tuple(bounds)


def binding_bound(bounds: Tuple[ResourceBound, ...]) -> ResourceBound | None:
    """The class that runs out first; None when nothing in the design uses any."""
    constrained = [bound for bound in bounds if bound.n_max is not None]
    return min(constrained, key=lambda b: b.n_max or 0) if constrained else None


DEVICE: Dict[str, Any] = {
    "part": DEVICE_PART,
    "card": DEVICE_CARD,
    "provenance": DEVICE_PROVENANCE,
    "totals": dict(DEVICE_TOTALS),
    "shell_overhead": SHELL_OVERHEAD_STATUS,
}
