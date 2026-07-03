"""SANA-FE per-component energy-breakdown record."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Dict


@dataclass
class SanafeEnergyBreakdown:
    """Per-component energy in joules.

    ``total_j`` is SANA-FE's reported total; :meth:`components_sum` cross-checks it
    (they agree to precision unless SANA-FE bills untracked overhead).
    """

    synapse_j: float
    dendrite_j: float
    soma_j: float
    network_j: float
    total_j: float

    def components_sum(self) -> float:
        return self.synapse_j + self.dendrite_j + self.soma_j + self.network_j

    def add(self, other: "SanafeEnergyBreakdown") -> "SanafeEnergyBreakdown":
        """Pure aggregation helper used by ``SanafeStepReport``."""
        return SanafeEnergyBreakdown(
            synapse_j=self.synapse_j + other.synapse_j,
            dendrite_j=self.dendrite_j + other.dendrite_j,
            soma_j=self.soma_j + other.soma_j,
            network_j=self.network_j + other.network_j,
            total_j=self.total_j + other.total_j,
        )

    @classmethod
    def zero(cls) -> "SanafeEnergyBreakdown":
        return cls(synapse_j=0.0, dendrite_j=0.0, soma_j=0.0,
                   network_j=0.0, total_j=0.0)

    @classmethod
    def from_sanafe_dict(cls, d: Dict[str, float]) -> "SanafeEnergyBreakdown":
        """Build from the ``results['energy']`` dict returned by ``chip.sim()``."""
        return cls(
            synapse_j=float(d.get("synapse", 0.0)),
            dendrite_j=float(d.get("dendrite", 0.0)),
            soma_j=float(d.get("soma", 0.0)),
            network_j=float(d.get("network", 0.0)),
            total_j=float(d.get("total", 0.0)),
        )

