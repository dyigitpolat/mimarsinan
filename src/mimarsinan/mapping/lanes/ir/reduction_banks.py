"""[U-L2] Banked reduction cores: a declared cost-only decision under position multiplexing (``reduction_core_banks``).

A pool lane's 0/1 reduction cores (``pool_cores``) each own their weights, one core per window group, so the one-copy
arm (W02), which serves a shared bank's instances in turn on one copy, holds one core set per window group, and a
resident program needs every one of them. Window groups of one pool that hold one matrix and read one producer grid
compute one law. Declared, each such set is the instances of one shared bank (one bank per distinct matrix and grid of
a lane, in the order the groups are met; a group of its own matrix is a bank of one instance), so W02 serves them in
turn on one copy, as it serves a conv's positions, and a resident program needs one copy per bank. Every core keeps
its matrix, threshold and law: outputs are the unbanked program's.

Both planes apply the one rule, keyed on what both hold: the window pattern (``pattern_key``: the matrix's extent and
its ones) and the producers' lane owners (whose grids the instances read), per pool op; banks are numbered after every
bank the program holds (``next_bank_id``). The builder's lane fusion banks the IR (``bank_ir_cores``), the shape-only
walk its records (``shape_walk.LaneWalk``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.ir.weight_bank import WeightBank
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError

#: The declaration key (platform constraints, mapping strategy).
KEY = "reduction_core_banks"


def declared_reduction_banks(platform: Mapping[str, Any]) -> bool:
    """Whether ``platform`` declares the decision: a bool, true only under ``position_multiplexing`` (the copy it
    shapes is the one-copy arm's); refused by name otherwise."""
    value = platform.get(KEY)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise PositionMultiplexingError(f"{KEY} must be a bool, got {value!r}")
    if value and not platform.get("position_multiplexing"):
        raise PositionMultiplexingError(f"{KEY} shares one copy of a lane's reduction cores between the window groups "
                                        "position multiplexing serves in turn: it needs position_multiplexing")
    return value


def pattern_key(rows: int, columns: int, ones: Iterable[Tuple[int, int]]) -> Tuple[int, int, Tuple]:
    """A 0/1 matrix as both planes hold it: its extent and its ones (row, column), sorted."""
    return int(rows), int(columns), tuple(sorted((int(r), int(c)) for r, c in ones))


def next_bank_id(bank_ids: Iterable[int]) -> int:
    """The first id after every bank the program holds."""
    return 1 + max((int(b) for b in bank_ids), default=-1)


def assign_banks(keys: Sequence[Hashable], first_id: int) -> List[int]:
    """Per item the id of its key's bank: one id per distinct key, from ``first_id``, in the order the keys are met."""
    ids: Dict[Hashable, int] = {}
    return [ids.setdefault(key, first_id + len(ids)) for key in keys]


class ReductionBankLawError(PositionMultiplexingError):
    """Instances of one reduction bank whose solved laws differ: one copy cannot serve them."""


def refuse_split_bank_laws(ir_graph: Any, laws: Mapping[int, Any], bank_ids: Iterable[int]) -> None:
    """Every instance of each bank in ``bank_ids`` holds one law (``laws``: core id to its solved law), as the copy
    that serves them states one; refused by name otherwise (the planes' key would have missed a grid)."""
    wanted = {int(b) for b in bank_ids}
    digests: Dict[int, set] = {}
    for node in ir_graph.nodes:
        bank = getattr(node, "weight_bank_id", None)
        if bank is not None and int(bank) in wanted:
            digests.setdefault(int(bank), set()).add(laws[int(node.id)].digest())
    split = sorted(b for b, seen in digests.items() if len(seen) > 1)
    if split:
        raise ReductionBankLawError(f"{KEY}: the instances of reduction banks {split[:4]} hold different laws, which "
                                    "one copy cannot serve")


def bank_ir_cores(ir_graph: Any, cores: Sequence[Any], owners_of: Callable[[Any], Hashable]) -> int:
    """Make ``cores`` (one pool op's reduction cores, owned 0/1 matrices) the instances of one bank per distinct
    pattern and producer owners (``owners_of``), banks added to ``ir_graph.weight_banks`` after every bank it holds.
    Each instance keeps its own sources, threshold and scales; its matrix becomes the bank's (the full column range).
    Returns the number of banks added."""
    matrices = [np.asarray(core.core_matrix) for core in cores]
    keys = [(pattern_key(m.shape[0], m.shape[1], np.argwhere(m != 0)), owners_of(core))
            for m, core in zip(matrices, cores)]
    ids = assign_banks(keys, next_bank_id(ir_graph.weight_banks))
    added = 0
    for core, matrix, bank_id in zip(cores, matrices, ids):
        if bank_id not in ir_graph.weight_banks:
            ir_graph.weight_banks[bank_id] = WeightBank(
                id=bank_id, core_matrix=matrix, activation_scale=core.activation_scale,
                parameter_scale=core.parameter_scale, input_activation_scale=core.input_activation_scale,
                perceptron_index=None, hardware_bias=None)
            added += 1
        core.core_matrix, core.weight_bank_id, core.weight_row_slice = None, bank_id, (0, int(matrix.shape[1]))
    return added


class WalkBanks:
    """[U-L2] The shape-only walk's side of the rule (``shape_walk.LaneWalk``): per pool op its cores' keys, numbered
    into banks at the walk's end; undeclared, every hook is a no-op."""

    def __init__(self, declared: bool) -> None:
        self.declared = bool(declared)
        self._ops: List[List[Tuple[int, Hashable, int, int]]] = []
        self._owner: Dict[int, int] = {}  # each inserted pool core's lane owner (a later pool's producer grid)

    def pool(self, mapping: Any, groups: Sequence[Tuple[List[int], List[int]]], windows: Sequence[Sequence[int]],
             flat: np.ndarray, outputs: np.ndarray, owner: int) -> np.ndarray:
        """Keep the keys of one pool op's cores (``groups``: per core its outputs and its rows over ``flat``;
        ``outputs``: the source of each pool output) and return ``outputs``."""
        if not self.declared:
            return outputs
        perceptron, index = mapping._sc_idx_to_perceptron_index, mapping._node_id_to_softcore_idx

        def owner_of(node: int) -> Any:
            found = perceptron.get(index.get(node))
            return self._owner.get(node) if found is None else found

        cores = []
        for members, rows in groups:
            node = int(outputs[members[0]].node_id)
            position = {k: r for r, k in enumerate(rows)}
            ones = [(position[k], column) for column, m in enumerate(members) for k in windows[m]]
            owners = frozenset(owner_of(int(flat[k].node_id)) for k in rows)
            cores.append((node, (pattern_key(len(rows), len(members), ones), owners), len(rows), len(members)))
            self._owner[node] = owner
        self._ops.append(cores)
        return outputs

    def apply(self, mapping: Any) -> None:
        """Number the kept cores' banks after every bank the walk recorded, in the order the pools were met; a per-axon
        split instance states no bank (the walk's rule for a conv's instances)."""
        first = next_bank_id(mapping._layout_weight_banks)
        for cores in self._ops:
            ids = assign_banks([key for _node, key, _rows, _columns in cores], first)
            for (node, _key, rows, columns), bank in zip(cores, ids):
                mapping._layout_weight_banks.setdefault(bank, (rows, columns))
                if mapping.rows_per_input == 1:
                    mapping._sc_idx_to_bank_id[mapping._node_id_to_softcore_idx[node]] = bank
            first = max(ids, default=first - 1) + 1
        if self._ops:
            mapping._next_bank_id = max(int(mapping._next_bank_id), first)
