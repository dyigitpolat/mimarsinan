"""W1 cascade equivalence certificate: value(pruned+compacted) must equal value(seeded-but-uncompacted) bit-exactly.

Certified per instance over N integer-valued probe batches through the real
deployed executor (identity hybrid build + ValueHybridCoreFlow, fp64); the
comparison is exact equality with zero tolerance, so preconditions require a
dyadic-grid-closed instance (see ``dyadic_grid``) and zero-preserving host
activations (see ``zero_preserving``). Transformer vehicles are explicitly
deferred until W0.5 lands; conv/FC mvm vehicles are in scope.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import IRGraph, IRSource
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.pruning.boundary_policy import assert_unified_ir_for_pruning
from mimarsinan.mapping.pruning.certificate.dyadic_grid import (
    DEFAULT_FRACTION_BITS,
    assert_dyadic_exactness_grid,
)
from mimarsinan.mapping.pruning.certificate.errors import (
    CascadeCertificateError,
    CascadeCertificatePreconditionError,
)
from mimarsinan.mapping.pruning.certificate.seed_reference import (
    SeedMasks,
    apply_admitted_seed_zeroing,
    check_shared_bank_union_rule,
    refuse_unusable_seeds,
)
from mimarsinan.mapping.pruning.certificate.zero_preserving import (
    assert_zero_preserving_preconditions,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import (
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
    _collect_initial_seeds,
)


@dataclass(frozen=True)
class CascadeEquivalenceCertificate:
    """Green result of one per-instance cascade equivalence certification."""

    batches: int
    batch_size: int
    input_size: int
    outputs_compared: int
    reference_cells: int
    pruned_cells: int
    bank_columns_checked: int
    max_abs_delta: float
    passed: bool

    def summary(self) -> str:
        return (
            f"cascade-equivalence: batches={self.batches}x{self.batch_size} "
            f"outputs={self.outputs_compared} cells {self.reference_cells}->"
            f"{self.pruned_cells} bank_cols_checked={self.bank_columns_checked} "
            f"max|delta|={self.max_abs_delta:.1e} passed={self.passed}"
        )


def _derive_input_size(ir_graph: IRGraph) -> int:
    max_index = -1
    for node in ir_graph.nodes:
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id == -2:
                max_index = max(max_index, int(src.index))
    if max_index < 0:
        raise CascadeCertificatePreconditionError(
            "graph has no model-input (-2) axons; nothing to probe."
        )
    return max_index + 1


def _physical_cells(hybrid_mapping) -> int:
    return sum(
        int(np.asarray(core.core_matrix).size)
        for stage in hybrid_mapping.stages
        if stage.kind == "neural"
        for core in stage.hard_core_mapping.cores
    )


def certify_cascade_equivalence(
    ir_graph: IRGraph,
    *,
    initial_pruned_per_node: SeedMasks | None = None,
    initial_pruned_per_bank: SeedMasks | None = None,
    zero_threshold: float = 1e-8,
    batches: int = 4,
    batch_size: int = 8,
    input_magnitude: int = 64,
    fraction_bits: int = DEFAULT_FRACTION_BITS,
    rng_seed: int = 0,
    spiking_mode: str = INERT_SPIKING_MODE,
    simulation_steps: int = 32,
    elimination_propagation: str = ELIMINATION_PROPAGATION_CASCADE,
) -> CascadeEquivalenceCertificate:
    """Certify that ``prune_ir_graph`` (+ deferred bank compaction in the
    identity build) preserved the program's value function bit-exactly.

    Reference = deep copy with the admitted seed masks applied as pure weight
    zeroing, no structural change; candidate = deep copy run through the real
    cascade. Both execute through the deployed value executor in fp64 on
    integer probe batches; any output bit difference raises. Never passes
    vacuously and never silently skips a precondition.
    """
    elimination_propagation = require_elimination_propagation(
        elimination_propagation
    )
    if not ir_graph.nodes:
        raise CascadeCertificatePreconditionError("empty IR graph; nothing to certify.")
    if 2.0 ** (-fraction_bits) <= zero_threshold:
        raise CascadeCertificatePreconditionError(
            f"dyadic grid 2^-{fraction_bits} is below zero_threshold="
            f"{zero_threshold}: sub-threshold nonzero weights would be "
            "eliminated inexactly."
        )
    assert_unified_ir_for_pruning(ir_graph)
    refuse_unusable_seeds(
        ir_graph, initial_pruned_per_node, initial_pruned_per_bank
    )
    assert_zero_preserving_preconditions(ir_graph)
    assert_dyadic_exactness_grid(ir_graph, fraction_bits=fraction_bits)
    input_size = _derive_input_size(ir_graph)

    exempt_rows, exempt_cols = _boundary_policy_exemptions(ir_graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        ir_graph, initial_pruned_per_node, initial_pruned_per_bank
    )
    fixpoint = compute_global_pruned_sets(
        ir_graph,
        zero_threshold=zero_threshold,
        initial_per_node=seed_per_node,
        initial_per_bank=seed_per_bank,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=elimination_propagation,
    )
    bank_columns_checked = check_shared_bank_union_rule(ir_graph, fixpoint)

    reference = copy.deepcopy(ir_graph)
    apply_admitted_seed_zeroing(
        reference, seed_per_node, seed_per_bank, exempt_rows, exempt_cols
    )
    candidate = copy.deepcopy(ir_graph)
    prune_ir_graph(
        candidate,
        zero_threshold=zero_threshold,
        initial_pruned_per_node=initial_pruned_per_node,
        initial_pruned_per_bank=initial_pruned_per_bank,
        spiking_mode=spiking_mode,
        simulation_steps=simulation_steps,
        elimination_propagation=elimination_propagation,
    )

    reference_hybrid = build_identity_hybrid_mapping(ir_graph=reference)
    candidate_hybrid = build_identity_hybrid_mapping(ir_graph=candidate)
    reference_flow = ValueHybridCoreFlow(reference_hybrid, dtype=torch.float64)
    candidate_flow = ValueHybridCoreFlow(candidate_hybrid, dtype=torch.float64)

    generator = torch.Generator().manual_seed(rng_seed)
    outputs_compared = 0
    for batch_index in range(batches):
        x = torch.randint(
            -input_magnitude, input_magnitude + 1,
            (batch_size, input_size), generator=generator,
        ).to(torch.float64)
        with torch.no_grad():
            want = reference_flow(x)
            got = candidate_flow(x)
        if want.shape != got.shape:
            raise CascadeCertificateError(
                f"cascade certificate TRIPPED: output shapes differ "
                f"(reference {tuple(want.shape)} vs pruned {tuple(got.shape)})."
            )
        if not torch.equal(want, got):
            delta = (got - want).abs()
            raise CascadeCertificateError(
                f"cascade certificate TRIPPED: batch {batch_index}: "
                f"{int((delta > 0).sum().item())}/{delta.numel()} output "
                f"values differ (max|delta|={float(delta.max().item()):.3e}); "
                "the pruned+compacted program is not value-identical to the "
                "seeded-but-uncompacted reference."
            )
        outputs_compared += int(want.numel())

    if outputs_compared == 0:
        raise CascadeCertificateError(
            "cascade certificate compared zero outputs; a certificate must "
            "never pass vacuously (batches and batch_size must be positive)."
        )
    return CascadeEquivalenceCertificate(
        batches=batches,
        batch_size=batch_size,
        input_size=input_size,
        outputs_compared=outputs_compared,
        reference_cells=_physical_cells(reference_hybrid),
        pruned_cells=_physical_cells(candidate_hybrid),
        bank_columns_checked=bank_columns_checked,
        max_abs_delta=0.0,
        passed=True,
    )
