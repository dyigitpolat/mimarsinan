"""The box's hypotheses at the model level: statement, assertion site, one-field off switch, exercise test.

Labels follow D17's twelve arms (``RA/04_experiment_portfolio.md:580``, X3a ``:1304``); ``R3`` (scoped
carry store) and ``R6`` (dependency-ordered walks) are rules of G5's pass projection
(``g5_residency_planner/u1_design.md`` section 5.1), kept as named extra arms, and so are ``R1`` (pins)
and ``R2`` (delivery delays), whose conjunction is H5-skew; R4 and R5 have no model-level counterpart
(the golden's encode is position-free and it has no per-hop re-timing). Two claims are checked per instance: relation ``packed`` (a packed multi-pass program
equals the identity single-pass program per neuron and window, X3a H0) and relation ``network`` (the
identity program equals the integer network N, the box's second equality). A hypothesis that is on is
honoured by the packer and checked by its site; an artifact that violates it is refused by name. The
off arm of a hypothesis switches exactly that one field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from . import kernels as K
from .program import STREAMED_LIF, Program
from .reference import network_claimed
from .schedule import Packed, producers


@dataclass(frozen=True)
class Hypothesis:
    label: str
    name: str
    statement: str
    site: str
    switch: str
    relation: str
    regimes: Tuple[str, ...]


_ALL = ("streamed_lif", "sync_lif", "sync_ttfs", "value")
HYPOTHESES: Tuple[Hypothesis, ...] = (
    Hypothesis("H1-theta", "Co-resident core-scalar parameter",
               "every fragment a hard core holds has the core's threshold (spiking) or shift (value)",
               "x3a_golden.hypotheses:check_coresident_scalar",
               "the coalescer ignores the core-scalar parameter; a core keeps its first fragment's", "packed", _ALL),
    Hypothesis("H1-latency", "Co-resident latency",
               "every fragment a hard core holds has the core's fused latency (streamed) or group (windowed)",
               "x3a_golden.hypotheses:check_coresident_latency",
               "the coalescer ignores latency; a core runs at its first fragment's", "packed", _ALL),
    Hypothesis("H2", "Window coverage",
               "every spike a neuron's in-segment producers emit in their windows arrives in its window",
               "x3a_golden.hypotheses:admit_network", "the coverage check is skipped", "network", ("streamed_lif",)),
    Hypothesis("H3", "No terminal overshoot", "the running staircase F(Q_t) never exceeds the terminal F(Q_T)",
               "x3a_golden.hypotheses:admit_network", "the overshoot check is skipped", "network", ("streamed_lif",)),
    Hypothesis("H4", "Chase completion", "the window count reaches clamp(F(Q_T), 0, T) (no burst debt left)",
               "x3a_golden.hypotheses:admit_network", "the deficit check is skipped", "network", ("streamed_lif",)),
    Hypothesis("H5-collapse", "Verbatim carry",
               "a pass cut carries the producer's raster (streamed LIF), emission train, level or value unchanged",
               "x3a_golden.machine:_Pass._kind",
               "a carried wire delivers its window count, re-encoded at the consumer", "packed",
               ("streamed_lif", "sync_ttfs")),
    Hypothesis("H5-skew", "Fused pass timing",
               "every pass runs its cores at the fused latencies and replays a carried raster on the fused "
               "absolute grid (G5 rules R1 pins and R2 delivery delays)",
               "x3a_golden.machine:_local_latency",
               "each pass runs its own ALAP latencies and reads a carried raster at the consumer's local cycle",
               "packed", ("streamed_lif",)),
    Hypothesis("H6", "Exact partial sums",
               "the parts of a fused neuron deliver exact integer partial charges to one membrane",
               "x3a_golden.machine:_streamed", "every part of a fused neuron is clipped to partial_sum_bits",
               "packed", _ALL),
    Hypothesis("H7", "Canonical event order",
               "a per-event core folds the events of a cycle in canonical rank order, whatever the axon slots",
               "x3a_golden.machine:_streamed", "per-event cores fold events in physical axon-slot order",
               "packed", ("streamed_lif",)),
    Hypothesis("H8", "Numeric range",
               "every accumulator's reachable bound fits its exact range (fp32 below 2**24; a saturating "
               "membrane never saturates where the staircase identity is claimed)",
               "x3a_golden.hypotheses:check_numeric_range",
               "fp32 accumulation and reachable saturation are admitted", "both", _ALL),
    Hypothesis("H9", "Dependency-ordered passes",
               "the pass partition is a DAG-monotone cut executed in a dependency order",
               "x3a_golden.hypotheses:check_pass_order", "units are dealt to passes and passes ordered at random",
               "packed", _ALL),
    Hypothesis("H10", "Residency integrity",
               "a pinned core's slot is never written by another core, so every pass runs its own programs",
               "x3a_golden.hypotheses:check_residency", "the slot allocator ignores the pinned slots", "packed", _ALL),
    Hypothesis("R1", "Pinned pass latencies",
               "every pass runs its cores at the fused segment's ALAP latencies (G5 rule R1)",
               "x3a_golden.machine:_local_latency",
               "each pass runs its own ALAP latencies; carried rasters keep the fused delivery delay", "packed",
               ("streamed_lif",)),
    Hypothesis("R2", "Delivery delays",
               "a carried raster is replayed at the fused delivery delay d = Lc - 1 - Lp (G5 rule R2)",
               "x3a_golden.machine:_streamed", "a carried raster is read at the consumer's local cycle (d = 0)",
               "packed", ("streamed_lif",)),
    Hypothesis("R6", "Dependency-ordered walks",
               "windowed executors (synchronized LIF and TTFS, value) evaluate a pass's cores in dependency order "
               "(G5 rule R6)",
               "x3a_golden.machine:_Pass.__init__",
               "windowed executors order cores by the stored streamed (ALAP) latency", "packed",
               ("sync_lif", "sync_ttfs", "value")),
    Hypothesis("R3", "Scoped carry store",
               "a segment entry re-encodes an earlier segment's window count; carried rasters are scoped to "
               "their producing segment (G5 rule R3)",
               "x3a_golden.machine:_Pass._kind", "a segment entry reads a stored raster of an earlier segment",
               "packed", ("streamed_lif",)),
)
LABELS = tuple(h.label for h in HYPOTHESES)
D17_ARMS = tuple(label for label in LABELS if not label.startswith("R"))
BY_LABEL = {h.label: h for h in HYPOTHESES}


def check_coresident_scalar(prog: Program, packed: Packed) -> List[str]:
    return [f"core {c.hid}" for c in packed.cores if len(set(c.frag_scalars)) > 1]


def check_coresident_latency(prog: Program, packed: Packed) -> List[str]:
    return [f"core {c.hid}" for c in packed.cores if len(set(c.frag_latencies)) > 1]


def check_pass_order(prog: Program, packed: Packed) -> List[str]:
    where = {h: i for i, (_, hids) in enumerate(packed.passes) for h in hids}
    prod, owner, bad = producers(prog, packed.cores), prog.owner(), []
    for c in packed.cores:
        for w in c.axons:
            nid = int(owner[w])
            src = prod.get((nid, w - prog.base(nid)))
            if src is not None and packed.cores[src].segment == c.segment and where[src] > where[c.hid]:
                bad.append(f"wire {w} pass {where[src]} > {where[c.hid]}")
        if c.partial:
            bad += [f"fused part {c.hid}" for lay, n, _ in c.slots if where[prod[(lay, n)]] != where[c.hid]]
    return bad


def check_residency(prog: Program, packed: Packed) -> List[str]:
    pinned = set(packed.pinned.values())
    return [f"core {h} in pinned slot {s}" for h, s in packed.slot_of.items() if s in pinned]


def bounds(prog: Program) -> Dict[int, np.ndarray]:
    return {n.nid: K.charge_bound(n.weights, n.bias, prog.decl) for n in prog.nodes if n.kind == "neural"}


def check_numeric_range(prog: Program, packed: Optional[Packed] = None) -> List[str]:
    if prog.decl.accumulator != "fp32":
        return []
    return [f"layer {nid}" for nid, b in bounds(prog).items() if (b >= K.FP32_EXACT).any()]


ARTIFACT_CHECKS = {"H1-theta": check_coresident_scalar, "H1-latency": check_coresident_latency,
                   "H8": check_numeric_range, "H9": check_pass_order, "H10": check_residency}


def validate(prog: Program, packed: Packed, off: FrozenSet[str]) -> List[str]:
    """Names of the hypotheses that are on and that the packed artifact violates (each refuses)."""
    return [label for label, check in ARTIFACT_CHECKS.items() if label not in off and check(prog, packed)]


def membrane_reachable(prog: Program) -> bool:
    bits = prog.decl.membrane_bits
    return bits is not None and any((b > (1 << (bits - 1)) - 1).any() for b in bounds(prog).values())


def admit_network(prog: Program, diag, off: FrozenSet[str]) -> Tuple[bool, List[str]]:
    """Whether relation ``network`` is claimed and admitted, and the named checks that refused it."""
    if not network_claimed(prog):
        return False, []
    refused = []
    if prog.decl.semantics == STREAMED_LIF:
        for label, key in (("H2", "coverage"), ("H3", "overshoot"), ("H4", "deficit")):
            if label not in off and any(v.any() for v in diag[key].values()):
                refused.append(label)
    if "H8" not in off and membrane_reachable(prog):
        refused.append("H8")
    return not refused, refused


def exercised(label: str, prog: Program, packed: Optional[Packed], stats, diag) -> bool:
    """Whether an instance exercises ``label``: the off arm could have changed its outputs."""
    d = prog.decl
    if label in ("H2", "H3", "H4"):
        key = {"H2": "coverage", "H3": "overshoot", "H4": "deficit"}[label]
        return network_claimed(prog) and any(v.any() for v in diag[key].values())
    if label == "H8":
        return bool(check_numeric_range(prog)) or (network_claimed(prog) and membrane_reachable(prog))
    if packed is None or stats is None:
        return False
    if label == "H1-theta":
        return bool(check_coresident_scalar(prog, packed))
    if label == "H1-latency":
        return bool(check_coresident_latency(prog, packed))
    if label in ("H5-collapse", "H5-skew", "R1", "R2"):
        return stats["carried_reads"] > 0
    if label == "H6":
        return any(g >= 0 for c in packed.cores for _, _, g in c.slots)
    if label == "H7":
        return d.granularity == "per_event" and any(c.axons != sorted(c.axons) for c in packed.cores)
    if label == "H9":
        return bool(check_pass_order(prog, packed))
    if label == "H10":
        return bool(check_residency(prog, packed))
    if label == "R3":
        return stats["stale_reads"] > 0
    if label == "R6":
        return stats["early_reads"] > 0
    raise KeyError(label)
