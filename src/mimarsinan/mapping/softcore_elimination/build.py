"""[W6b] Aggregate softcore facts into the per-group, per-arm report."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.pruning.elimination_ledger.arm_runs import (
    EliminationArms,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
)
from mimarsinan.mapping.softcore_elimination.facts import (
    BankFacts,
    InstanceFacts,
    SoftcoreFacts,
    facts_from_masks,
    facts_from_pruning_result,
)
from mimarsinan.mapping.softcore_elimination.naming import (
    softcore_group_name,
    softcore_layer_name,
)
from mimarsinan.mapping.softcore_elimination.report import (
    EliminationView,
    SoftcoreEliminationReport,
)
from mimarsinan.mapping.softcore_elimination.types import (
    GEOMETRY_PRE_ELIMINATION,
    SHARED_GROUP_LABEL,
    UNMAPPED_GROUP_LABEL,
    GroupElimination,
    StorageElimination,
    aggregate_groups,
    aggregate_storage,
)

# The arm label used when only the realized elimination is known (a stored,
# already-pruned IR carries no weaker-arm kill sets to difference against).
REALIZED_ARM = "realized"


def _grouped(
    instances: Sequence[InstanceFacts], key: Callable[[str], str]
) -> tuple[GroupElimination, ...]:
    buckets: dict[str, list[InstanceFacts]] = defaultdict(list)
    for inst in instances:
        buckets[key(inst.name)].append(inst)
    rows = [
        _group_row(name, members) for name, members in sorted(buckets.items())
    ]
    return tuple(rows)


def _group_row(name: str, members: Sequence[InstanceFacts]) -> GroupElimination:
    dims = {(m.axons, m.neurons) for m in members}
    axons, neurons = dims.pop() if len(dims) == 1 else (None, None)
    return GroupElimination(
        group=name,
        instances=len(members),
        axons=axons,
        neurons=neurons,
        rows_eliminated=sum(m.rows_eliminated for m in members),
        cols_eliminated=sum(m.cols_eliminated for m in members),
        cells=sum(m.cells for m in members),
        surviving=sum(m.surviving for m in members),
        layers=tuple(sorted({softcore_layer_name(m.name) for m in members})),
    )


def _storage_group(bank: BankFacts) -> str:
    """The group a shared bank belongs to: the one every sharer maps to."""
    groups = {softcore_group_name(name) for name in bank.sharers}
    if not groups:
        return UNMAPPED_GROUP_LABEL
    if len(groups) > 1:
        return SHARED_GROUP_LABEL
    return groups.pop()


def _storage_rows(facts: SoftcoreFacts) -> tuple[StorageElimination, ...]:
    """Physical weight storage per group: banks once + unshared owned matrices."""
    banks: dict[str, list[BankFacts]] = defaultdict(list)
    for bank in facts.banks:
        banks[_storage_group(bank)].append(bank)
    owned: dict[str, list[InstanceFacts]] = defaultdict(list)
    for inst in facts.instances:
        if inst.weight_bank_id is None:
            owned[softcore_group_name(inst.name)].append(inst)

    rows = []
    for name in sorted(set(banks) | set(owned)):
        group_banks = banks.get(name, ())
        group_owned = owned.get(name, ())
        rows.append(StorageElimination(
            group=name,
            banks=len(group_banks),
            bank_cells_before=sum(b.cells for b in group_banks),
            bank_cells_after=sum(b.surviving for b in group_banks),
            owned_matrices=len(group_owned),
            owned_cells_before=sum(o.cells for o in group_owned),
            owned_cells_after=sum(o.surviving for o in group_owned),
        ))
    return tuple(rows)


def build_view(arm: str, facts: SoftcoreFacts) -> EliminationView:
    """One arm's complete view: group rows, layer rows, both totals."""
    groups = _grouped(facts.instances, softcore_group_name)
    layers = _grouped(facts.instances, softcore_layer_name)
    storage = _storage_rows(facts)
    return EliminationView(
        arm=arm,
        groups=groups,
        layers=layers,
        total=aggregate_groups(groups),
        storage=storage,
        storage_total=aggregate_storage(storage),
    )


def report_from_arms(
    ir_graph: IRGraph, arms: EliminationArms
) -> SoftcoreEliminationReport:
    """Build the per-run report at the soft-core mapping seam.

    ``ir_graph`` must be the UNCOMPACTED graph the arms were run on: every
    mapped softcore is then present at its pre-elimination geometry, so the
    masked / closure / cascade columns share one denominator and their
    differences are exactly the increments closure coupling and emergent
    propagation bought.
    """
    results: Mapping[str, GlobalPruningResult] = arms.results_by_arm()
    views = {
        arm: build_view(arm, facts_from_pruning_result(ir_graph, result))
        for arm, result in results.items()
    }
    return SoftcoreEliminationReport(
        deployed_arm=arms.mode,
        views=views,
        geometry=GEOMETRY_PRE_ELIMINATION,
    )


def report_from_pruned_ir_graph(
    ir_graph: IRGraph, *, geometry: str = GEOMETRY_PRE_ELIMINATION
) -> SoftcoreEliminationReport:
    """Reconstruct the report from a stored, already-pruned IR graph.

    Only the REALIZED elimination is recoverable this way — the weaker arms
    left no trace in the graph — so the report carries a single ``realized``
    arm. Cores the liveness pass deleted are gone from the graph and therefore
    absent from the denominator, which understates elimination.
    """
    facts = facts_from_masks(ir_graph, geometry=geometry)
    return SoftcoreEliminationReport(
        deployed_arm=REALIZED_ARM,
        views={REALIZED_ARM: build_view(REALIZED_ARM, facts)},
        geometry=geometry,
    )
