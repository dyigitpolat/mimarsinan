"""Lane fusion on the IR (WI-5e): the planned host ops leave the program and live on their producing cores.

A clamp op is removed and its readers read the cores (the clamp is the law's ``lo = 0``, solved later). A fused add
is removed: each accumulator column of its producing cores gains one operand row (zero crossbar weights, the other
input's element as its source) and the add's readers read the cores. A producer-side reduction replaces its
producer's cores by reduced cores (``reduce_cores``: each pool output stacks its window instances on the producer's
weights, one rounding). Any other average pool becomes reduction cores (0/1 weights over each window, ``theta`` the
window size) that requantize onto the pool lane's grid. Every role is keyed on the plan's call sites (one per
call of a module, ``call_sites``), so the IR and the model absorb the same ops; a planned op the IR cannot fuse is
refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir.source import encode_ir_sources
from mimarsinan.mapping.lanes.forms import PoolForm, pool_form
from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles, planned_call_sites
from mimarsinan.mapping.lanes.ir.operand_rows import insert_operand_rows
from mimarsinan.mapping.lanes.ir.pool_cores import PoolLaneError, pool_lane_cores
from mimarsinan.mapping.lanes.ir.reduce_cores import reduced_lane_cores, refuse_unreducible
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from mimarsinan.mapping.lanes.ir.rewire import rewire_removed
from mimarsinan.mapping.lanes.plan import LanePlan
from mimarsinan.mapping.latency.ir import IRLatency

#: The attribute a core carries once a fused add gave its columns operand rows: the row per column (``-1``: none).
OPERAND_ROWS = "lane_operand_rows"
#: The attribute a pool lane's reduction core carries: its lane's owner key (a perceptron's is its index).
LANE_OWNER = "lane_owner"
#: The attribute an IR graph carries once its lanes are fused: the fusion report (for records and the census).
LANE_REPORT = "lane_fusion_report"


@dataclass
class LaneFusion:
    """What fusion did, for the step report and the lane census."""

    clamps: int = 0
    adds: int = 0
    pools: int = 0
    pool_cores: int = 0
    operand_rows: int = 0
    lane_ops: Dict[str, int] = field(default_factory=lambda: {"clamp": 0, "add": 0, "pool": 0})
    pool_owners: Dict[str, int] = field(default_factory=dict)
    pool_forms: Dict[int, PoolForm] = field(default_factory=dict)
    reductions: int = 0
    reduced_cores: int = 0
    #: perceptron index -> the window its reduction sums (for the model's stamp).
    reduce_forms: Dict[int, PoolForm] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"clamps": self.clamps, "adds": self.adds, "pools": self.pools, "pool_cores": self.pool_cores,
                "operand_rows": self.operand_rows, "lane_ops": dict(self.lane_ops),
                "pool_owners": dict(self.pool_owners), "reductions": self.reductions,
                "reduced_cores": self.reduced_cores}


def _readers(ir_graph: IRGraph) -> Dict[int, set]:
    """{producer id: ids of the nodes reading it} (``-1`` for the program output)."""
    readers: Dict[int, set] = {}
    for node in ir_graph.nodes:
        encoded = encode_ir_sources(node.input_sources)
        ids = encoded[2] if isinstance(encoded, tuple) else np.array([int(s.node_id) for s in node.input_sources.flat])
        for producer in np.unique(ids[ids >= 0]).tolist():
            readers.setdefault(int(producer), set()).add(int(node.id))
    for source in np.asarray(ir_graph.output_sources).flat:
        if isinstance(source, IRSource) and source.node_id >= 0:
            readers.setdefault(int(source.node_id), set()).add(-1)
    return readers


class _Fuser:
    def __init__(self, ir_graph: IRGraph, plan: LanePlan, max_axons: Optional[int], max_neurons: Optional[int]):
        self.ir, self.plan = ir_graph, plan
        self.cores: Dict[int, NeuralCore] = {int(n.id): n for n in ir_graph.nodes if isinstance(n, NeuralCore)}
        # A reduction core never outgrows the declared core, or (no geometry declared) the program's widest core.
        self.max_axons = max_axons or max((c.get_input_count() for c in self.cores.values()), default=None)
        self.max_neurons = max_neurons or max((c.get_output_count() for c in self.cores.values()), default=None)
        self.readers = _readers(ir_graph)
        self.replace: Dict[int, np.ndarray] = {}  # removed op id -> the sources its output elements become
        self.operands: Dict[int, Dict[int, IRSource]] = {}  # core id -> {column: operand source}
        self.inserted: Dict[int, List[NeuralCore]] = {}  # removed pool op id -> its reduction cores
        self.removed_cores: set = set()  # producer cores a reduction replaced
        self.next_id = 1 + max((int(n.id) for n in ir_graph.nodes), default=0)
        self.report = LaneFusion()

    def weight_dtype(self):
        """The integer dtype the quantized program's matrices carry (a reduction core's 0/1 weights take it)."""
        for core in self.cores.values():
            matrix = core.get_core_matrix(self.ir)
            if np.issubdtype(np.asarray(matrix).dtype, np.integer):
                return np.asarray(matrix).dtype
        return np.int8

    def resolved(self, op: ComputeOp) -> np.ndarray:
        out = []
        for source in op.input_sources.flatten():
            hit = self.replace.get(int(source.node_id)) if source.node_id >= 0 else None
            out.append(hit[int(source.index)] if hit is not None else source)
        return np.array(out, dtype=object)

    def _on_cores(self, sources: np.ndarray, op: ComputeOp, role: str, *, off: bool = True) -> None:
        """Every element is a core's output; an off element (a pruned column, constant 0) passes a clamp or a pool
        as 0, but an add's accumulator needs a core column to hold its operand (``off=False``)."""
        bad = sorted({int(s.node_id) for s in sources
                      if int(s.node_id) not in self.cores and not (off and s.is_off())})
        if bad:
            raise LaneFusionError(f"lane {role} {op.name!r} reads nodes {bad[:4]} that are not cores (host or input "
                                  f"values): the planned lane has no on-chip producer")

    def _effective_readers(self, producer: int) -> set:
        """The nodes reading ``producer``'s values, through every op already fused (its readers read them now)."""
        found = set()
        for reader in self.readers.get(producer, set()):
            found |= self._effective_readers(reader) if reader in self.replace else {reader}
        return found

    def _exclusive(self, sources: np.ndarray, op: ComputeOp) -> None:
        for producer in sorted({int(s.node_id) for s in sources if int(s.node_id) >= 0}):
            others = self._effective_readers(producer) - {int(op.id)}
            if others:
                raise LaneFusionError(f"core {self.cores[producer].name!r} is read by nodes {sorted(others)[:3]} "
                                      f"beside its lane {op.name!r}: the lane's rounding would change their value")

    def _refuse_host_fed(self, acc: np.ndarray, op: ComputeOp) -> None:
        """An accumulator core with an armed entry grid reads host values: the executors snap its input columns on
        that grid per core, which an operand row read on its producer's grid cannot share (the plan keeps such an
        add on the host, so reaching here means the IR and the plan disagree)."""
        for core_id in sorted({int(s.node_id) for s in acc}):
            grid = getattr(self.cores[core_id], "boundary_grid", None)
            if grid is not None and grid.armed:
                raise LaneFusionError(f"lane add {op.name!r} accumulates into core {self.cores[core_id].name!r}, "
                                      "which reads host values on an entry grid: a fused add needs an on-chip "
                                      "accumulator")

    def clamp(self, op: ComputeOp) -> None:
        sources = self.resolved(op)
        self._on_cores(sources, op, "clamp")
        self._exclusive(sources, op)
        self.replace[int(op.id)] = sources
        self.report.clamps += 1
        self.report.lane_ops["clamp"] += int(sources.size)

    def add(self, op: ComputeOp, operand_index: int) -> None:
        shapes = [tuple(s) for s in op.params.get("input_shapes") or ()]
        if len(shapes) != 2 or shapes[0] != shapes[1]:
            raise LaneFusionError(f"lane add {op.name!r} has operand shapes {shapes}: a fused add takes two equal "
                                  f"shapes")
        sources = self.resolved(op)
        size = sources.size // 2
        halves = (sources[:size], sources[size:])
        acc, operand = halves[1 - operand_index], halves[operand_index]
        self._on_cores(acc, op, "add", off=False)
        self._on_cores(operand, op, "add operand")
        self._exclusive(acc, op)
        self._refuse_host_fed(acc, op)
        for a, b in zip(acc, operand):
            columns = self.operands.setdefault(int(a.node_id), {})
            if int(a.index) in columns:
                raise LaneFusionError(f"core {self.cores[int(a.node_id)].name!r} column {a.index} would take two "
                                      f"operands (a lane fuses one add)")
            columns[int(a.index)] = b
        self.replace[int(op.id)] = acc
        self.report.adds += 1
        self.report.lane_ops["add"] += int(size)

    def pool(self, op: ComputeOp, owner: int) -> None:
        sources = self.resolved(op)
        self._on_cores(sources, op, "pool")
        cores, outputs, form = pool_lane_cores(op, sources, owner=owner, first_id=self.next_id,
                                               max_axons=self.max_axons, max_neurons=self.max_neurons,
                                               dtype=self.weight_dtype())
        for core in cores:
            setattr(core, LANE_OWNER, owner)
            self.cores[int(core.id)] = core
        self.next_id += len(cores)
        self.inserted[int(op.id)] = cores
        self.replace[int(op.id)] = outputs
        self.report.pools += 1
        self.report.pool_cores += len(cores)
        # One lane op per pool output the lane requantizes: the window sums run on the 0/1 reduction cores (priced
        # as their crossbar cells), so the summed inputs are no lane work (ng45_os_lane: P needs no lane logic).
        self.report.lane_ops["pool"] += int(np.asarray(outputs).size)
        self.report.pool_owners[str(op.name)] = owner
        self.report.pool_forms[owner] = form


    def reduce(self, op: ComputeOp) -> None:
        sources = self.resolved(op)
        self._on_cores(sources, op, "reduction", off=False)
        self._exclusive(sources, op)
        producers = sorted({int(s.node_id) for s in sources})
        for core_id in producers:
            refuse_unreducible(self.cores[core_id], op)
            if getattr(self.cores[core_id], LANE_OWNER, None) is not None or core_id in self.operands:
                raise LaneFusionError(f"reduction {op.name!r} reads core {self.cores[core_id].name!r}, which another "
                                      "lane already rewrote")
        perceptrons = {int(self.cores[c].perceptron_index) for c in producers}  # type: ignore[arg-type]
        if len(perceptrons) != 1:
            raise LaneFusionError(f"reduction {op.name!r} reads perceptrons {sorted(perceptrons)}: one producer")
        form = pool_form(op.params.get("module"), tuple(op.input_shape or ()))  # pyright: ignore[reportArgumentType]
        if form is None:
            raise PoolLaneError(f"reduction {op.name!r} over input {op.input_shape}: no fixed window with a constant "
                                "divisor")
        cores, outputs = reduced_lane_cores(op, sources, self.cores, self.ir, form=form, first_id=self.next_id)
        for core in cores:
            self.cores[int(core.id)] = core
        self.next_id += len(cores)
        self.inserted[int(op.id)] = cores
        self.removed_cores.update(producers)
        self.replace[int(op.id)] = outputs
        self.report.reductions += 1
        self.report.reduced_cores += len(cores)
        self.report.lane_ops["pool"] += int(np.asarray(outputs).size)  # the requantized pool outputs, as pool()
        self.report.reduce_forms[perceptrons.pop()] = form


def fuse_ir_lanes(ir_graph: IRGraph, plan: LanePlan, *, first_owner: int, max_axons: Optional[int] = None,
                  max_neurons: Optional[int] = None) -> LaneFusion:
    """Fuse ``plan``'s ops into ``ir_graph`` in place; pool lanes take owner keys from ``first_owner`` on."""
    sites = planned_call_sites(plan, first_owner)
    roles, carried = call_site_roles(ir_graph, sites)
    fuser = _Fuser(ir_graph, plan, max_axons, max_neurons)
    removed = set()
    for node in list(ir_graph.nodes):
        if not isinstance(node, ComputeOp):
            continue
        role = roles.get(int(node.id))
        if role is None:
            continue
        kind, argument = role
        if kind == "clamp":
            fuser.clamp(node)
        elif kind == "add":
            fuser.add(node, int(argument))  # pyright: ignore[reportArgumentType] -- the add role carries an int
        elif kind == "reduce":
            fuser.reduce(node)
        else:
            fuser.pool(node, int(argument))  # pyright: ignore[reportArgumentType] -- the pool role carries an int
        removed.add(int(node.id))
    if carried != len(sites):
        raise LaneFusionError(f"the lane plan absorbs {len(sites)} host ops but the IR carries {carried} of "
                              f"them: the IR was mapped from another model")
    fuser.report.operand_rows = insert_operand_rows(ir_graph, fuser.operands, OPERAND_ROWS)
    nodes = []
    for node in ir_graph.nodes:
        if int(node.id) in removed:
            nodes.extend(fuser.inserted.get(int(node.id), ()))
        elif int(node.id) not in fuser.removed_cores:
            nodes.append(node)
    ir_graph.nodes = nodes
    if fuser.removed_cores:
        ir_graph._cleanup_orphan_weight_banks()  # a reduced producer's bank, if no core reads it now
    rewire_removed(ir_graph, fuser.replace)
    IRLatency(ir_graph).calculate()
    return fuser.report
