"""[wsm] The weight-programming boundary, measured: a pure read over packed programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WeightProgrammingReport:
    """Weight-programming cost of one deployed forward, per the packed program.

    Today's scheduled builds rebuild a fresh pool per pass, so every placed
    instance programs its weights (``params_programmed`` counts matrix params
    per placement; bias registers are neurons-sized and excluded).
    ``params_unique`` is the weight-stationary ideal (each distinct bank /
    owned matrix programmed once); ``reuse_factor = unique / programmed``
    (1.0 = ideal; 1/N = an N-instance bank fully re-programmed).
    """

    neural_stages: int
    programming_events: int
    params_programmed: int
    params_unique: int

    @property
    def reuse_factor(self) -> float:
        if self.params_programmed <= 0:
            return 1.0
        return self.params_unique / self.params_programmed

    def summary(self) -> str:
        return (
            f"stages={self.neural_stages} events={self.programming_events} "
            f"params_programmed={self.params_programmed} "
            f"unique={self.params_unique} "
            f"reuse_factor={self.reuse_factor:.2f} "
            f"(1.0 = each param programmed once)"
        )


def weight_programming_report(hybrid_mapping: Any) -> WeightProgrammingReport:
    """Compute the report from a ``HybridHardCoreMapping`` (placement dicts)."""
    neural_stages = 0
    events = 0
    programmed = 0
    unique_bank_params: dict[int, int] = {}
    owned_params = 0

    for stage in hybrid_mapping.stages:
        if stage.kind != "neural" or stage.hard_core_mapping is None:
            continue
        neural_stages += 1
        if getattr(stage, "schedule_weights_resident", False):
            # [wsm V2] verified bank-clustered pass: weights stayed resident
            # from the previous pass — zero programming cost.
            continue
        segment = stage.hard_core_mapping
        bank_matrices = getattr(segment, "weight_banks", {}) or {}
        for placements in segment.soft_core_placements_per_hard_core:
            for placement in placements:
                area = int(placement["axons"]) * int(placement["neurons"])
                events += 1
                programmed += area
                bank_id = placement.get("weight_bank_id")
                if bank_id is None:
                    owned_params += area
                elif bank_id not in unique_bank_params:
                    matrix = bank_matrices.get(bank_id)
                    unique_bank_params[bank_id] = (
                        int(matrix.size) if matrix is not None else area
                    )

    return WeightProgrammingReport(
        neural_stages=neural_stages,
        programming_events=events,
        params_programmed=programmed,
        params_unique=owned_params + sum(unique_bank_params.values()),
    )
