"""Dry-run placement feasibility: run the REAL hard-core packer, no GPU, no sim.

The SOUND lower bound (:func:`estimate_cores_needed`) admits any config whose
IDEAL diagonal packing fits the budget. But the greedy packer cannot place
softcores from different threshold groups on one hard core (a hard core, once
claimed by a softcore's ``threshold_group_id``, only accepts that group —
``canonical._read_threshold_group``), and ``threshold_group_id`` is the softcore's
PERCEPTRON INDEX (structural, weight-independent), so a config can pass the lower
bound yet crash late in the pack with ``RuntimeError("No more hard cores
available")`` once per-perceptron fragmentation is accounted for.

:func:`dryrun_pack_feasible` runs the ACTUAL hybrid packer on the IR graph (CPU,
sub-second, deterministic — the build is weight-independent) and returns a
definitive verdict: the EARLY, diagnosable catch the lower bound cannot give.
Because the threshold grouping is structural, an untrained model's dry-run is
bit-identical to the trained deployment's packing — so this is an exact
feasibility oracle, with no false-rejection risk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_SEGMENT_RE = re.compile(r"segment '([^']+)'")


@dataclass(frozen=True)
class PackFeasibility:
    """Definitive placement verdict from running the real hybrid packer.

    ``feasible`` is True when the packer placed the whole IR graph; ``hard_cores``
    then counts the placed hard cores (None when infeasible). On failure,
    ``overflowing_segment`` names the segment the packer ran out of cores in (parsed
    from the diagnostic) and ``error`` carries the packer's message.
    """

    feasible: bool
    hard_cores: int | None
    overflowing_segment: str | None
    error: str | None


def dryrun_pack_feasible(
    ir_graph, platform_constraints: Mapping[str, Any]
) -> PackFeasibility:
    """Run the real hybrid hard-core packer on ``ir_graph`` and report feasibility.

    No GPU, no simulation, no trained weights: the packing decisions depend only
    on the IR's structural shape (axon/neuron counts, perceptron-indexed threshold
    groups) and the declared core budget, so this is the same packer the deployment
    runs — invoked early to reject a doomed config before any GPU is claimed.
    Resolves the same :class:`MappingStrategy` (coalesce / split / schedule
    permissions) the mapping step uses, so a scheduling-feasible config is admitted.
    """
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import (
        ChipCapabilities,
        MappingStrategy,
    )

    strategy = MappingStrategy.resolve(
        ChipCapabilities.from_platform_constraints(platform_constraints)
    )
    try:
        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=platform_constraints["cores"],
            strategy=strategy,
        )
    except RuntimeError as exc:
        message = str(exc)
        match = _SEGMENT_RE.search(message)
        return PackFeasibility(
            feasible=False,
            hard_cores=None,
            overflowing_segment=match.group(1) if match else None,
            error=message,
        )

    hard_cores = sum(
        len(segment.cores) for segment in hybrid_mapping.get_neural_segments()
    )
    return PackFeasibility(
        feasible=True,
        hard_cores=int(hard_cores),
        overflowing_segment=None,
        error=None,
    )
