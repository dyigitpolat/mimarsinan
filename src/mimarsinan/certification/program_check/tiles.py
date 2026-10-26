"""[W15] The deployment's tiles against the identity program's: one sweep over every placement of the deployment.

Per placement it compares the neuron block's fan-in, summed per source (``view.canonical``), with the identity
program's columns of the same IR node, so a tile on the wrong core or columns, a dropped, duplicated or re-signed
partial sum, a permuted wire or a bias stamped differently is a difference named by node, column and source. It also
reads, in the same pass, the core singletons (H1-theta), the stamped laws, the fused cores (H6) and the accumulation
facts H8 bounds; the column coverage makes the placement records a bijection onto the identity's neurons."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from mimarsinan.certification.program_check.reads import training_module
from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.view import (
    OFF, Canonical, Placed, ProgramView, canonical, decode, first_difference, restrict, same,
)
from mimarsinan.mapping.platform.core_residency import CORE_SINGLETON_PROPERTIES
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources

#: Singletons every executor reads from the hard core whatever the target declares (one register per core).
EXECUTOR_READ = ("threshold",)


@dataclass
class CoreFacts:
    """What the sweep learned of one deployed core: its live rows (a nonzero weight in a used column), the largest
    per-neuron weighted fan-in magnitude, the largest weight and input code, and whether its weights, bias and
    threshold are integral."""

    live: np.ndarray
    magnitude: float = 0.0
    max_weight: float = 0.0
    max_level: float = 0.0
    placed: bool = False
    integral_weights: bool = True
    integral_bias: bool = True
    integral_threshold: bool = True
    nonnegative: bool = True
    theta: float = 1.0
    law: Any = None


@dataclass
class SweepFacts:
    cores: Dict[Tuple[int, int], CoreFacts] = field(default_factory=dict)
    width: Dict[int, int] = field(default_factory=dict)
    identity_cores: Dict[Tuple[int, int], CoreFacts] = field(default_factory=dict)


def _where(p: Placed, view: ProgramView) -> str:
    unit = view.units[p.unit]
    return f"unit {unit.index} ({unit.stage.name}) core {p.core} node {p.node} columns {p.col0}-{p.col0 + p.cols - 1}"


def _source(value: int) -> str:
    node, index = decode(value)
    return {-3: "always-on axon", -4: "hardware bias"}.get(node, f"node {node}[{index}]")


def _block(cv, lo: int, hi: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The executed weights (the bank block on the HCM's bank fast path) and hardware bias of columns ``[lo, hi)``."""
    bias = getattr(cv.core, "hardware_bias", None)
    return cv.weights()[:cv.used_axons, lo:hi], (None if bias is None else np.asarray(bias)[lo:hi])


def _facts(cv, block_all, bias_all, row_levels, include_bias: bool) -> CoreFacts:
    live = np.any(block_all != 0, axis=1) & (cv.codes != OFF)
    used = np.asarray(block_all, dtype=np.float64)[cv.codes != OFF]
    levels = row_levels[cv.codes != OFF][:, None]
    # A row whose input range is undeclared (NaN) taints the bound only where it carries a weight.
    mags = np.where(used != 0, np.abs(used) * levels, 0.0).sum(axis=0) if used.size else np.zeros(1)
    bias = None if bias_all is None else np.asarray(bias_all[:cv.used_neurons], dtype=np.float64)
    if include_bias and bias is not None and bias.size:
        mags = mags + np.abs(bias)
    theta = float(getattr(cv.core, "threshold", 1.0) or 0.0)
    weighted = levels[:, 0][np.any(used != 0, axis=1)] if used.size else np.zeros(0)
    return CoreFacts(live=live, magnitude=float(np.max(mags, initial=0.0)), theta=theta, placed=bool(cv.placements),
                     max_weight=float(np.abs(used).max(initial=0.0)),
                     max_level=float(np.nanmax(weighted, initial=0.0)) if weighted.size else 0.0,
                     law=getattr(cv.core, "readout_law", None),
                     integral_weights=bool(np.array_equal(used, np.round(used))),
                     integral_bias=bias is None or bool(np.array_equal(bias, np.round(bias))),
                     integral_threshold=theta == round(theta),
                     nonnegative=bool((used >= 0).all()) and (bias is None or bool((bias >= 0).all())))


def _singletons(core) -> Tuple[str, ...]:
    constrained = getattr(core, "constrained_properties", None) or ()
    return tuple(p.name for p in CORE_SINGLETON_PROPERTIES if p.name in EXECUTOR_READ or p.name in constrained)


def _law_columns(core, lo: int, hi: int):
    law = getattr(core, "readout_law", None)
    return None if law is None else law.take_columns(np.arange(lo, hi))


def _laws_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))


def sweep(delta: ProgramView, iota: ProgramView, results: Dict[str, ConditionResult], *,
          row_levels: Callable[[Any, Any], np.ndarray], include_bias: bool = True) -> SweepFacts:
    """Compare every deployed placement with the identity's columns; fill ``tiles``, ``H1-theta``, ``H6`` and
    ``laws``; return the per-core facts H8, H8-staircase, H3 and the timing conditions read."""
    tiles, theta, h6, laws = (results[c] for c in ("tiles", "H1-theta", "H6", "laws"))
    facts = SweepFacts()
    ident: Dict[int, Placed] = {}
    for node, placed in iota.placements.items():
        if len(placed) != 1:
            tiles.add(Violation("tiles", "identity_placement", f"node {node}",
                                f"the identity program places it {len(placed)} times"))
        ident[node] = placed[0]
        facts.width[node] = placed[0].cols
    for u, unit in enumerate(iota.units):
        for h, cv in enumerate(unit.cores):
            grid, bias = _block(cv, 0, cv.used_neurons)
            facts.identity_cores[(u, h)] = _facts(cv, grid, bias, row_levels(unit, cv), include_bias)
    pending = {node: len(placed) for node, placed in delta.placements.items()}
    cache: Dict[int, Canonical] = {}
    coverage = {node: np.zeros(width, dtype=np.int64) for node, width in facts.width.items()}

    def identity_block(node: int) -> Canonical:
        if node not in cache:
            p = ident[node]
            cv = iota.units[p.unit].cores[p.core]
            block, bias = _block(cv, p.neuron_offset, p.neuron_offset + p.cols)
            cache[node] = canonical(cv.codes, block, bias)
        return cache[node]

    for u, unit in enumerate(delta.units):
        for h, cv in enumerate(unit.cores):
            grid, bias_all = _block(cv, 0, cv.used_neurons)
            facts.cores[(u, h)] = _facts(cv, grid, bias_all, row_levels(unit, cv), include_bias)
            fused = getattr(cv.core, "fused_component_axons", None)
            if fused is not None:
                h6.checked += 1
                if int(sum(fused)) != int(cv.core.axons_per_core):   # metadata: the fit and the costs read it
                    tiles.add(Violation("tiles", "fused_axons", f"unit {unit.index} core {h}",
                                        f"components {list(fused)} sum to {sum(fused)}, not its "
                                        f"{cv.core.axons_per_core} axons (metadata no executor reads)"))
            for p in cv.placements:
                tiles.checked += 1
                where = _where(p, delta)
                if p.node not in ident:
                    tiles.add(Violation("tiles", "foreign_node", where, "the identity program holds no such node"))
                    continue
                lo, hi = p.neuron_offset, p.neuron_offset + p.cols
                if hi > cv.used_neurons or p.col0 + p.cols > facts.width[p.node]:
                    tiles.add(Violation("tiles", "placement_extent", where,
                                        f"core columns {lo}-{hi - 1} of {cv.used_neurons} used, node width "
                                        f"{facts.width[p.node]}"))
                    continue
                coverage[p.node][p.col0:p.col0 + p.cols] += 1
                mine = restrict(canonical(cv.codes, grid[:, lo:hi], None if bias_all is None else
                                          np.asarray(bias_all)[lo:hi]), 0, p.cols)
                theirs = restrict(identity_block(p.node), p.col0, p.col0 + p.cols)
                if not same(mine, theirs):
                    diff = first_difference(mine, theirs)
                    detail = ("equal sums, different order" if diff is None else
                              f"column {p.col0 + diff[1]} from {_source(diff[0])}: {diff[2]!r} deployed, "
                              f"{diff[3]!r} in the identity program")
                    tiles.add(Violation("tiles", "fan_in", where, detail))
                    if fused is not None:
                        h6.add(Violation("H6", "partial_sums", where, detail))
                ip = ident[p.node]
                icore = iota.units[ip.unit].cores[ip.core].core
                for name in _singletons(cv.core):
                    theta.checked += 1
                    prop = next(q for q in CORE_SINGLETON_PROPERTIES if q.name == name)
                    if not prop.compare(getattr(cv.core, name, None), getattr(icore, name, None)):
                        theta.add(Violation("H1-theta", name, where,
                                            f"the core stores {getattr(cv.core, name, None)!r}, the node's "
                                            f"identity core {getattr(icore, name, None)!r}"))
                mine_law = _law_columns(cv.core, lo, hi)
                their_law = _law_columns(icore, ip.neuron_offset + p.col0, ip.neuron_offset + p.col0 + p.cols)
                if mine_law is not None or their_law is not None:
                    laws.checked += 1
                    if not _laws_equal(mine_law, their_law):
                        laws.add(Violation("laws", "stamped_law", where,
                                           "the core's law columns differ from the node's solved law"))
                pending[p.node] -= 1
                if pending[p.node] == 0:
                    cache.pop(p.node, None)
    for node, cov in coverage.items():
        if (cov != 1).any():
            bad = np.flatnonzero(cov != 1)
            tiles.add(Violation("tiles", "coverage", f"node {node}",
                                f"{len(bad)} column(s) placed {sorted(set(cov[bad].tolist()))} times "
                                f"(first {int(bad[0])}); every identity neuron is placed exactly once"))
    for u, h, k in delta.unplaced:
        unit = delta.units[u]
        tiles.add(Violation("tiles", "unplaced_column", f"unit {unit.index} ({unit.stage.name}) core {h}",
                            f"used column {k} belongs to no placement record (or to two)"))
    return facts


#: The program's per-node boundary maps the stage loop's decode and encode read.
NODE_MAPS = ("node_activation_scales", "node_input_activation_scales", "node_buffer_scales", "node_output_shifts")


def _values_equal(a, b) -> bool:
    def flat(v):
        return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float64)
    try:
        x, y = flat(a), flat(b)
    except (TypeError, ValueError):
        return a is b
    return x.shape == y.shape and bool(np.array_equal(x, y))


def _same_value(a, b) -> bool:
    """A host op's parameter in the deployment and the identity program: the same object, or a module of the same
    type, configuration and state (a deep copy of the program copies its host modules)."""
    if a is b:
        return True
    if isinstance(a, torch.nn.Module) or isinstance(b, torch.nn.Module):
        if type(a) is not type(b) or a.extra_repr() != b.extra_repr():
            return False
        sa, sb = a.state_dict(), b.state_dict()
        return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    if isinstance(a, (np.ndarray, torch.Tensor)) or isinstance(b, (np.ndarray, torch.Tensor)):
        return _values_equal(a, b)
    try:
        return bool(a == b)
    except (TypeError, ValueError, RuntimeError):  # an incomparable value is not the same parameter
        return False


def wiring(delta: ProgramView, iota: ProgramView, program: Any, identity: Any, width: Dict[int, int],
           tiles: ConditionResult) -> None:
    """Stage outputs, the program outputs and the host stages: every output slice is written by the neurons its node
    owns, the network reads the identity's outputs, and every host stage is the identity's host op."""
    for unit in delta.units:
        out = np.asarray(unit.stage.hard_core_mapping.output_sources).flatten().tolist()
        for node, offset, size in [(int(s.node_id), int(s.offset), int(s.size)) for s in unit.stage.output_map]:
            tiles.checked += 1
            where = f"unit {unit.index} ({unit.stage.name}) output slice of node {node}"
            if node in width and size != width[node]:
                tiles.add(Violation("tiles", "output_width", where, f"size {size}, node width {width[node]}"))
            for span in compress_spike_sources(out[offset:offset + size]):
                n = int(span.length)
                want = np.arange(int(span.dst_start), int(span.dst_start) + n)
                if span.kind != "core" or not 0 <= int(span.src_core) < len(unit.cores):
                    tiles.add(Violation("tiles", "output_source", where, f"positions {want[:3].tolist()}... read "
                                        f"a {span.kind} source, not the node's neurons"))
                    continue
                cv = unit.cores[int(span.src_core)]
                idx = np.arange(int(span.src_start), int(span.src_start) + n)
                ok = idx < cv.used_neurons
                got_n = np.where(ok, cv.node_of[np.where(ok, idx, 0)], -1)
                got_c = np.where(ok, cv.col_of[np.where(ok, idx, 0)], -1)
                if not ((got_n == node) & (got_c == want)).all():
                    tiles.add(Violation("tiles", "output_source", where, f"positions {want[:3].tolist()}... read "
                                        f"core {span.src_core} neurons that the placements give to other columns"))
    for name in NODE_MAPS:
        mine, theirs = getattr(program, name, None) or {}, getattr(identity, name, None) or {}
        tiles.checked += 1
        differ = sorted(int(k) for k in set(mine) | set(theirs)
                        if k not in mine or k not in theirs or not _values_equal(mine[k], theirs[k]))
        if differ:
            tiles.add(Violation("tiles", "node_scales", f"program {name}", f"node(s) {differ[:6]} differ from the "
                                "identity program's (the boundary decode and encode read them)"))
    pairs = [[(int(s.node_id), int(s.index)) for s in np.asarray(p.output_sources).flatten()]
             for p in (program, identity)]
    tiles.checked += 1
    if pairs[0] != pairs[1]:
        tiles.add(Violation("tiles", "program_outputs", "program", "the network reads other buffers than the "
                            "identity program's outputs"))
    theirs = {int(st.compute_op.id): st.compute_op for _, _, st in iota.hosts if st.compute_op is not None}
    for _, _, stage in delta.hosts:
        op = stage.compute_op
        tiles.checked += 1
        other = theirs.get(int(getattr(op, "id", -1)))
        where = f"host stage {stage.name}"
        if other is None:
            tiles.add(Violation("tiles", "host_op", where, "no identity host op of its id"))
            continue
        src = [(int(s.node_id), int(s.index)) for s in np.asarray(op.input_sources).flatten()]
        ref = [(int(s.node_id), int(s.index)) for s in np.asarray(other.input_sources).flatten()]
        same_params = set(op.params) == set(other.params) and all(_same_value(op.params[k], other.params[k])
                                                                  for k in op.params)
        if src != ref or not same_params or type(op) is not type(other):
            tiles.add(Violation("tiles", "host_op", where, "its sources, module or parameters differ from the "
                                "identity's host op (Lemma 7 needs the same function of the same inputs)"))
        mode = training_module(op.params or {})
        if mode is not None:
            tiles.add(Violation("tiles", "host_training", where, f"holds a {mode} in training mode: the host op is "
                                "batch-dependent, no fixed function of its inputs (Lemma 7)"))
