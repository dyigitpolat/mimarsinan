"""[W6c] Aggregate softcore facts into the per-group, per-arm report.

The pipeline is deliberately ordered so that IDENTITY comes first and naming
last:

1. bucket the instances by their STRUCTURAL mapped-layer key
   (:mod:`...identity`) -- one bucket per mapped source layer, whatever the
   nodes happen to be called;
2. label the buckets (:mod:`...labels`) -- display only;
3. collapse repeated-container rows for presentation, over rows that are
   already structurally correct.

A workload whose layers are literally named ``0``, ``1``, ``2`` therefore gets
one correct row per layer, and no naming outcome can merge two layers or split
one.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

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
    facts_from_pruning_result,
)
from mimarsinan.mapping.softcore_elimination.identity import MappedLayerKey
from mimarsinan.mapping.softcore_elimination.labels import (
    collapse_repeats,
    display_labels,
)
from mimarsinan.mapping.softcore_elimination.mask_facts import (
    facts_from_masks,
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


def _by_layer_key(
    instances: Sequence[InstanceFacts],
) -> dict[MappedLayerKey, list[InstanceFacts]]:
    """One bucket per mapped source layer — the structural row set."""
    buckets: dict[MappedLayerKey, list[InstanceFacts]] = defaultdict(list)
    for inst in instances:
        buckets[inst.layer_key].append(inst)
    return dict(buckets)


def _layer_row(label: str, members: Sequence[InstanceFacts]) -> GroupElimination:
    dims = {(m.axons, m.neurons) for m in members}
    axons, neurons = dims.pop() if len(dims) == 1 else (None, None)
    return GroupElimination(
        group=label,
        instances=len(members),
        axons=axons,
        neurons=neurons,
        rows_eliminated=sum(m.rows_eliminated for m in members),
        cols_eliminated=sum(m.cols_eliminated for m in members),
        cells=sum(m.cells for m in members),
        surviving=sum(m.surviving for m in members),
        layers=(label,),
    )


def _storage_rows(
    facts: SoftcoreFacts, group_of: Mapping[MappedLayerKey, str]
) -> tuple[StorageElimination, ...]:
    """Physical weight storage per group: banks once + unshared owned matrices."""
    banks: dict[str, list[BankFacts]] = defaultdict(list)
    for bank in facts.banks:
        banks[_storage_group(bank, group_of)].append(bank)
    owned: dict[str, list[InstanceFacts]] = defaultdict(list)
    for inst in facts.instances:
        if inst.weight_bank_id is None:
            owned[group_of[inst.layer_key]].append(inst)

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


def _storage_group(
    bank: BankFacts, group_of: Mapping[MappedLayerKey, str]
) -> str:
    """The group a shared bank belongs to: the one every sharer maps to."""
    groups = {group_of[key] for key in bank.sharer_keys if key in group_of}
    if not groups:
        return UNMAPPED_GROUP_LABEL
    if len(groups) > 1:
        return SHARED_GROUP_LABEL
    return groups.pop()


def build_view(arm: str, facts: SoftcoreFacts) -> EliminationView:
    """One arm's complete view: group rows, layer rows, both totals."""
    buckets = _by_layer_key(facts.instances)
    labels = display_labels(buckets)
    layers = tuple(sorted(
        (_layer_row(labels[key], members) for key, members in buckets.items()),
        key=lambda row: row.group,
    ))
    groups = collapse_repeats(layers)
    # The storage view is bucketed by the SAME collapsed label, so the two
    # halves of the table line up row for row.
    collapsed_of = {
        layer: row.group for row in groups for layer in row.layers
    }
    group_of = {key: collapsed_of[label] for key, label in labels.items()}
    storage = _storage_rows(facts, group_of)
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
