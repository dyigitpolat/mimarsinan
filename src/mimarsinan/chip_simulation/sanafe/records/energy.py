"""SANA-FE run-record dataclasses.

These records carry SANA-FE's *full* output for a single sample — per-tile
and per-core energy breakdowns, latency (``sim_time``), NoC packet counts,
optional per-neuron spike + potential traces, and the spike-count subset
that overlaps with HCM's ``spike_recorder.RunRecord``.

Why our own record types
------------------------
HCM's ``RunRecord`` is a spike-count-only shape sized for the diff-based
parity gate.  SANA-FE produces a richer, multi-dimensional output; bolting
the rich fields onto ``RunRecord`` would mix two simulators' contracts.
Instead, we keep the rich record native to SANA-FE and expose a single
lossless projection — :meth:`SanafeRunRecord.to_hcm_subset` — to bridge
back to the HCM shape when the parity gate runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Dict


@dataclass
class SanafeEnergyBreakdown:
    """Per-component energy in joules.

    SANA-FE reports ``total`` independently of the components so we store
    both: ``total_j`` is what SANA-FE returns; :meth:`components_sum` is
    the cross-check.  In practice they agree to numerical precision, but
    if SANA-FE ever bills overhead components we don't track, the gap is
    visible rather than papered over.
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

