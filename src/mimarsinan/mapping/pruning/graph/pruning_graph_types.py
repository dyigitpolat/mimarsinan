from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set

@dataclass
class GlobalPruningResult:
    """Per-node and per-bank pruned row/column sets after global fixpoint."""

    pruned_rows_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_rows_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_bank: Dict[int, Set[int]] = field(default_factory=dict)

