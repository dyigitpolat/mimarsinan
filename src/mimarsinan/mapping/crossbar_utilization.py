"""Crossbar occupancy of one deployed program — the IMC resource ledger."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

UTILIZATION_RECORD_FILENAME = "crossbar_utilization.json"


def _ratio(used: int, total: int) -> float:
    return (used / total) if total else 0.0


@dataclass(frozen=True)
class CoreOccupancy:
    """One allocated crossbar: physical geometry vs the rows/columns actually used.

    On an IMC substrate a row/column is reclaimed only when eliminated whole, so
    ``*_used`` is the reclaimable-structure measure that ``*_physical`` bounds.
    """

    axons_used: int
    axons_physical: int
    neurons_used: int
    neurons_physical: int
    unusable_space: int

    @property
    def cells_physical(self) -> int:
        return self.axons_physical * self.neurons_physical

    @property
    def cells_used(self) -> int:
        return self.axons_used * self.neurons_used

    @property
    def occupancy(self) -> float:
        return _ratio(self.cells_used, self.cells_physical)

    @classmethod
    def from_hard_core(cls, core: Any) -> "CoreOccupancy":
        """Read a ``HardCore``: used = geometry minus the capacity still available."""
        axons = int(getattr(core, "axons_per_core", 0) or 0)
        neurons = int(getattr(core, "neurons_per_core", 0) or 0)
        return cls(
            axons_used=axons - int(getattr(core, "available_axons", 0) or 0),
            axons_physical=axons,
            neurons_used=neurons - int(getattr(core, "available_neurons", 0) or 0),
            neurons_physical=neurons,
            unusable_space=int(getattr(core, "unusable_space", 0) or 0),
        )


@dataclass(frozen=True)
class CrossbarUtilizationReport:
    """Allocated crossbars and how full they are, for one deployed program.

    ``macs`` counts one multiply-accumulate per programmed cell per forward;
    ``programming_bits`` (cells x weight bits) is the write-cost energy proxy and
    is None when the platform does not declare a weight width.
    """

    cores: tuple[CoreOccupancy, ...]
    weight_bits: int | None

    @classmethod
    def from_hard_cores(
        cls, cores: Iterable[Any], *, weight_bits: int | None = None
    ) -> "CrossbarUtilizationReport":
        return cls(
            cores=tuple(CoreOccupancy.from_hard_core(c) for c in cores),
            weight_bits=None if weight_bits is None else int(weight_bits),
        )

    @classmethod
    def from_hybrid_mapping(
        cls, hybrid_mapping: Any, *, weight_bits: int | None = None
    ) -> "CrossbarUtilizationReport":
        """Aggregate every neural stage's packed cores across the whole program."""
        cores: list[Any] = []
        for stage in getattr(hybrid_mapping, "stages", ()):
            hcm = getattr(stage, "hard_core_mapping", None)
            if hcm is not None:
                cores.extend(hcm.cores)
        return cls.from_hard_cores(cores, weight_bits=weight_bits)

    @property
    def cores_allocated(self) -> int:
        return len(self.cores)

    @property
    def axons_used(self) -> int:
        return sum(c.axons_used for c in self.cores)

    @property
    def axons_physical(self) -> int:
        return sum(c.axons_physical for c in self.cores)

    @property
    def neurons_used(self) -> int:
        return sum(c.neurons_used for c in self.cores)

    @property
    def neurons_physical(self) -> int:
        return sum(c.neurons_physical for c in self.cores)

    @property
    def cells_used(self) -> int:
        return sum(c.cells_used for c in self.cores)

    @property
    def cells_physical(self) -> int:
        return sum(c.cells_physical for c in self.cores)

    @property
    def unusable_space(self) -> int:
        return sum(c.unusable_space for c in self.cores)

    @property
    def axon_utilization(self) -> float:
        return _ratio(self.axons_used, self.axons_physical)

    @property
    def neuron_utilization(self) -> float:
        return _ratio(self.neurons_used, self.neurons_physical)

    @property
    def cell_occupancy(self) -> float:
        return _ratio(self.cells_used, self.cells_physical)

    @property
    def macs(self) -> int:
        return self.cells_used

    @property
    def programming_bits(self) -> int | None:
        if self.weight_bits is None:
            return None
        return self.cells_used * self.weight_bits

    def to_dict(self) -> dict[str, Any]:
        """Flat scalar record — one row of the experiment table."""
        return {
            "cores_allocated": self.cores_allocated,
            "axons_used": self.axons_used,
            "axons_physical": self.axons_physical,
            "axon_utilization": self.axon_utilization,
            "neurons_used": self.neurons_used,
            "neurons_physical": self.neurons_physical,
            "neuron_utilization": self.neuron_utilization,
            "cells_used": self.cells_used,
            "cells_physical": self.cells_physical,
            "cell_occupancy": self.cell_occupancy,
            "unusable_space": self.unusable_space,
            "macs": self.macs,
            "weight_bits": self.weight_bits,
            "programming_bits": self.programming_bits,
        }


def write_utilization_record(
    report: CrossbarUtilizationReport, run_directory: str
) -> str:
    """Serialize the flat record as JSON into the run directory; returns the path."""
    path = os.path.join(run_directory, UTILIZATION_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


def summarize_utilization(report: CrossbarUtilizationReport) -> str:
    """One-line run log, mirroring the weight-programming report's shape."""
    bits = "n/a" if report.programming_bits is None else f"{report.programming_bits:.3e}"
    return (
        f"[Crossbar] cores={report.cores_allocated} "
        f"axons={report.axons_used}/{report.axons_physical} "
        f"({report.axon_utilization:.3f}) "
        f"neurons={report.neurons_used}/{report.neurons_physical} "
        f"({report.neuron_utilization:.3f}) "
        f"occupancy={report.cell_occupancy:.4f} "
        f"macs={report.macs:.3e} prog_bits={bits}"
    )
