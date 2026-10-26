"""[W15] Proposition 2's conditions, read off the identity program and the IR: exact staircases (H8-staircase), the
static parts of window coverage (H2) and no terminal overshoot (H3), chase completion (H4, data-dependent), the
identity program as the IR's network (compaction drops only dead rows and columns), the seam currencies (calculus
11.2), the synchronized TTFS half-step preconditions (P3, P4), and the stamped laws against the laws the solver gives
the IR (when the calibrated ranges are declared)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from mimarsinan.certification.program_check.declaration import CheckDeclaration
from mimarsinan.certification.program_check.report import (
    ESTABLISHED, NOT_STATIC, VACUOUS, ConditionResult, Violation,
)
from mimarsinan.certification.program_check.tiles import SweepFacts
from mimarsinan.certification.program_check.timing import IdentityTiming
from mimarsinan.mapping.export.law_solver import solve_ir_laws
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.packing.softcore.compaction import _ir_keep_indices
from mimarsinan.mapping.support.activation_scales import scalar_node_scale

#: The executors' staircase kernels take the integer charge (closure fix MJ4, sign-off P1 items 1 to 3:
#: ``models/spiking/wire_semantics.lif_window_count`` and the synchronized TTFS lattice ladder), so H8-staircase
#: reduces, per program, to the integer lattice of every weight, bias and threshold.
STAIRCASE_KERNELS = "lif_window_count (synchronized LIF count), the TTFS lattice ladder (chip_simulation/ttfs)"


def check_staircase(facts: SweepFacts, result: ConditionResult) -> None:
    """H8-staircase on the identity program: every weight, bias and threshold integral; no saturating membrane is a
    declarable form on this tree, so the saturation clause is vacuous."""
    for (u, h), core_facts in facts.identity_cores.items():
        result.checked += 1
        if not (core_facts.integral_weights and core_facts.integral_bias and core_facts.integral_threshold):
            result.add(Violation("H8-staircase", "off_lattice", f"identity unit {u} core {h}",
                                 f"threshold {core_facts.theta!r} or a weight or bias is not integral, so the "
                                 "staircase is not taken on the integer charge"))
    result.note = f"kernels: {STAIRCASE_KERNELS}; saturation clause vacuous (no saturating membrane declared)"


def check_coverage(ident: IdentityTiming, result: ConditionResult) -> None:
    """H2 (streamed LIF): every in-segment edge of the identity program at a one-cycle gap covers the producer's
    window by construction; a wider gap leaves window coverage to the data (the sampled run)."""
    result.checked = 1
    if ident.gaps:
        result.status = NOT_STATIC
        widest = max(g for _, _, g in ident.gaps)
        result.note = (f"{len(ident.gaps)} in-segment edge(s) at latency gaps up to {widest}: whether a producer "
                       "fires in its last cycles is data-dependent; stays with the sampled run")
    else:
        result.note = "every in-segment edge of the identity program sits at a one-cycle gap"


def check_overshoot(facts: SweepFacts, result: ConditionResult) -> None:
    """H3: a neuron whose fan-in weights and bias are all nonnegative has a monotone charge, so its running
    staircase never exceeds the terminal one; any other neuron leaves H3 to the data (the sampled run)."""
    signed = sum(not f.nonnegative for f in facts.identity_cores.values())
    result.checked = len(facts.identity_cores)
    if signed:
        result.status = NOT_STATIC
        result.note = (f"{signed} identity core(s) carry a negative weight or bias: overshoot is data-dependent; "
                       "stays with the sampled run")
    else:
        result.note = "every identity neuron's charge is monotone (nonnegative fan-in)"


def chase_completion(result: ConditionResult) -> None:
    result.status = NOT_STATIC
    result.note = ("chase completion (every burst debt drains before the window ends) depends on the arrival "
                   "pattern of the inputs; no input-independent predicate implies it; stays with the sampled run")


def check_identity_network(ir_graph: Any, result: ConditionResult) -> None:
    """The identity program computes the IR's network only where compaction drops what is dead. A dropped column is
    dead when its weights and bias are all zero (its output is identically zero); the identity program reads a
    dropped column as an off axon, so a dropped column that is not dead must be no network output and be read by no
    kept row with a nonzero weight; a dropped row may carry a nonzero weight on a kept column only where its source is
    an off axon or a dead column. A dropped row whose source is a host op's output leaves the condition not static
    (the host op decides whether that output is zero)."""
    cores = [n for n in getattr(ir_graph, "nodes", ()) if isinstance(n, NeuralCore)]
    masked = [c for c in cores if getattr(c, "pruned_row_mask", None) is not None
              and getattr(c, "pruned_col_mask", None) is not None]
    if not masked:
        result.status, result.note = VACUOUS, "no IR core carries compaction masks"
        return
    keep = {int(c.id): _ir_keep_indices(c) for c in cores}
    dead, live_dropped = set(), set()
    for core in masked:
        rows, cols = keep[int(core.id)]
        matrix = np.asarray(core.get_core_matrix(ir_graph))
        bias = np.zeros(matrix.shape[1]) if core.hardware_bias is None else np.asarray(core.hardware_bias)
        for c in sorted(set(range(matrix.shape[1])) - set(cols)):
            (dead if not np.any(matrix[:, c] != 0) and bias[c] == 0 else live_dropped).add((int(core.id), c))
    hosts = {int(n.id) for n in getattr(ir_graph, "nodes", ()) if not isinstance(n, NeuralCore)}
    undecided = 0
    outputs = {(int(s.node_id), int(s.index)) for s in np.asarray(ir_graph.output_sources).flatten()}
    for node, column in sorted(live_dropped & outputs):
        result.add(Violation("N-identity", "dropped_output", f"IR node {node}",
                             f"column {column} is a network output, not dead, and compaction drops it"))
    for core in cores:
        result.checked += 1
        rows, cols = keep[int(core.id)]
        if not cols:
            continue
        sources = [(int(s.node_id), int(s.index)) for s in core.input_sources.flatten()]
        matrix = None
        dropped = set(range(len(sources))) - set(rows)
        for r, src in enumerate(sources):
            reads_live_drop = src in live_dropped and r not in dropped
            drops_live_row = r in dropped and src[0] != -1 and src not in dead
            if not (reads_live_drop or drops_live_row):
                continue
            if matrix is None:
                matrix = np.asarray(core.get_core_matrix(ir_graph))
            if not np.any(matrix[r, cols] != 0):
                continue
            if drops_live_row and src[0] in hosts:
                undecided += 1      # a host output's zero is the host op's property (pruning's own certificate)
                continue
            rule = "dropped_read_column" if reads_live_drop else "dropped_live_row"
            result.add(Violation("N-identity", rule, f"IR node {core.id} row {r}",
                                 f"reads node {src[0]}[{src[1]}] with a nonzero weight, but compaction "
                                 + ("drops that column though it is not dead" if reads_live_drop else
                                    "drops the row though its source is live")))
    if undecided and not result.violation_count:
        result.status = NOT_STATIC
        result.note = (f"{undecided} dropped row(s) carry a nonzero weight from a host op's output: whether that output "
                       "is zero is the host op's property (the IR pruning's zero-preservation certificate), not decided "
                       "statically; stays with the sampled run")


def check_solved_laws(ir_graph: Any, law_inputs: Optional[Mapping[str, Any]], result: ConditionResult) -> None:
    """Every IR core's stamped law equals the law ``solve_ir_laws`` gives it from the declared spec and calibrated
    ranges (``law_inputs``: ``ranges``, ``spec``); without them the re-solve is not run, and the deployed columns are
    held to the IR's stamped laws only (the ``laws`` condition)."""
    carried = [n for n in getattr(ir_graph, "nodes", ()) if getattr(n, "readout_law", None) is not None]
    if not carried:
        return
    if law_inputs is None:
        result.note += "; the re-solve against the solver did not run (no calibrated ranges declared)"
        return
    solved = solve_ir_laws(ir_graph, law_inputs["ranges"], law_inputs["spec"])
    for node in carried:
        if not isinstance(node, NeuralCore):
            continue
        result.checked += 1
        if getattr(node, "readout_law").digest() != solved[int(node.id)].digest():
            result.add(Violation("laws", "solved_law", f"IR node {node.id}", "the stamped law is not the law the "
                                 "solver gives this core"))
    if result.status == ESTABLISHED:
        result.note += f"; {len(carried)} IR laws re-solved and equal"


def check_seams(ir_graph: Any, identity: Any, result: ConditionResult) -> None:
    """Currency coherence at every spiking seam (calculus 11.2): a core that reads host-op outputs must re-encode them
    at its trained entry currency. Statically decided where every seam is at unit gauges (the core's entry currency
    ``input_activation_scale`` and the gauge each host source emits, the identity's ``node_activation_scales``, all
    1): coherent by construction. A seam with another gauge depends on how its host op owns its scale domain (a
    ``ScaleNormalizingWrapper`` emits at its own armed gauge, a heterogeneous join folds per-source scales into the
    weights, calculus 16.6), which the program does not keep; the install certificate
    (``verify_boundary_currency_coherence``) decides it, so the condition stays with the sampled run there."""
    hosts = {int(n.id) for n in getattr(ir_graph, "nodes", ()) if isinstance(n, ComputeOp)}
    scales = getattr(identity, "node_activation_scales", None) or {}
    non_unit = []
    for node in getattr(ir_graph, "nodes", ()):
        if not isinstance(node, NeuralCore):
            continue
        deps = sorted({int(s.node_id) for s in node.input_sources.flatten() if int(s.node_id) in hosts})
        if not deps:
            continue
        result.checked += 1
        gauges = [scalar_node_scale(scales.get(d, 1.0)) for d in deps]
        stamped = float(np.mean(np.asarray(node.input_activation_scale, dtype=np.float64)))
        if stamped != 1.0 or any(g != 1.0 for g in gauges):
            non_unit.append(int(node.id))
    if non_unit:
        result.status = NOT_STATIC
        result.note = (f"{len(non_unit)} of {result.checked} seam reader(s) at a non-unit gauge (first IR node "
                       f"{non_unit[0]}): coherence is the install certificate's (calculus 11.2, 16.6); stays with "
                       "the sampled run")
    else:
        result.note = f"{result.checked} seam reader(s), every one at unit gauges: coherent by construction"


def check_half_step(decl: CheckDeclaration, result: ConditionResult) -> None:
    """Synchronized TTFS's P3 and P4: vacuous when no half-step is folded and the comparator half-step is off;
    otherwise whether the half-step is folded once and carried by the bias lattice is the conversion's (the
    floor-ceil convention's bias compensation, ``SoftCoreMappingStep``), so the condition stays with the sampled run."""
    result.checked = 1
    if decl.half_step_folded or decl.comparator_half_step:
        result.status = NOT_STATIC
        result.note = ("a half-step is " + ("folded into the biases" if decl.half_step_folded else
                                            "carried by the compare ladder") + ": P3 and P4 bind and are the "
                       "conversion's; stays with the sampled run")
    else:
        result.note = "no half-step folded and the comparator half-step off: P3 and P4 vacuous"
