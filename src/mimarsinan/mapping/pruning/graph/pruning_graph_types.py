from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set

@dataclass
class GlobalPruningResult:
    """Per-node and per-bank pruned row/column sets after global fixpoint.

    ``fixpoint_iterations`` counts the global refresh sweeps executed
    (including the final quiescent one): 0 for the masked arm, 1 for the
    closure arm, and the actual sweep count for the cascade fixpoint.
    """

    pruned_rows_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_rows_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    fixpoint_iterations: int = 0

