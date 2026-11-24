"""The lane plan (WI-5e): which host ops of the mapper graph a lane absorbs, decided once, on the model.

The placement rule is the toggle's (arith_toggle.md §3.5): an op of a declared lane class runs on chip when its
producers are on chip, and a lane ends at one rounding. A perceptron's lane absorbs, in order, a fused add (the
perceptron the add's only reader of, the other input any on-chip value: the operand) or an average pool it alone
feeds (the producer-side reduction, §3.4: the pool sums the perceptron's accumulators over each window, one
rounding for the pooled value), then a ReLU (the lane's last value's only reader: the clamp); any other average
pool over an on-chip value is a lane of its own (a requantizing reduction). A fused add also needs its accumulator
on chip: a perceptron that reads a host value (the network input or a host op, a segment entry) snaps its rows on
its entry grid, which a core-fed operand row does not share, so its add stays on the host (exact there). A
constant-zero pad over an on-chip value is on-chip wiring (``forms.zero_pad_form``): its padded elements are off
sources, which add 0, so an add whose operand is a zero-padded on-chip value takes its on-chip addition and its sum is
rounded once (branch tcad/residual-sum-rounding). An add class E would put on chip that stays on the host is noted
with its holder and why (``host_adds``), for the declared state's refusal (``declared_lanes``). Every decision reads
the graph and the declaration only, so Boundary Quantization, Readout Law and Soft Core Mapping recompute the same
plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from mimarsinan.mapping.export.output_law import WIRING_MAPPERS
from mimarsinan.mapping.lanes.declaration import ADD, CLAMP, POOL, LaneDeclaration
from mimarsinan.mapping.lanes.forms import is_add_form, is_clamp_form, is_pool_form, pool_form, zero_pad_form
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper


@dataclass
class PerceptronLane:
    """A perceptron's lane: its holder, then optionally a fused add (``operand_index`` names the add input that is
    the operand) or a reduction (an average pool the holder alone feeds), and a clamp; the lane's value is the
    output of :attr:`end`."""

    holder: Any
    add: Optional[ComputeOpMapper] = None
    operand_index: int = -1
    clamp: Optional[ComputeOpMapper] = None
    reduce: Optional[ComputeOpMapper] = None

    @property
    def end(self) -> Any:
        return self.clamp or self.add or self.reduce or self.holder

    @property
    def unsigned(self) -> bool:
        return self.clamp is not None


@dataclass
class PoolLane:
    """An average pool over an on-chip value: a requantizing reduction lane, unsigned over an unsigned producer."""

    pool: ComputeOpMapper
    unsigned: bool


@dataclass
class HostAdd:
    """An add class E would put on chip that stays on the host: ``holder``, the perceptron it alone reads (whose lane
    it would end, so whose lane grid would be the sum's), and why it stays."""

    add: ComputeOpMapper
    holder: Any
    reason: str


@dataclass
class LanePlan:
    declaration: LaneDeclaration
    perceptron_lanes: List[PerceptronLane] = field(default_factory=list)
    pool_lanes: List[PoolLane] = field(default_factory=list)
    #: Constant-zero pads over on-chip values: on-chip wiring, absorbed as the lanes' ops are.
    pads: List[ComputeOpMapper] = field(default_factory=list)
    host_adds: List[HostAdd] = field(default_factory=list)

    def absorbed(self) -> FrozenSet[int]:
        """``id`` of every host-op mapper a lane absorbs (an on-chip op, never a segment boundary)."""
        ids = {id(m) for lane in self.perceptron_lanes for m in (lane.add, lane.reduce, lane.clamp) if m is not None}
        return frozenset(ids | {id(lane.pool) for lane in self.pool_lanes} | {id(pad) for pad in self.pads})

    def lane_of(self, holder) -> Optional[PerceptronLane]:
        return next((lane for lane in self.perceptron_lanes if lane.holder is holder), None)

    def counts(self) -> Dict[str, int]:
        """The lane ops by kind, and the absorbed pads where there are any (absent at 0, so a plan without a pad
        reports as it did)."""
        counts = {"clamp": sum(lane.clamp is not None for lane in self.perceptron_lanes),
                  "add": sum(lane.add is not None for lane in self.perceptron_lanes), "pool": len(self.pool_lanes),
                  "reduce": sum(lane.reduce is not None for lane in self.perceptron_lanes)}
        if self.pads:
            counts["pad"] = len(self.pads)
        return counts


def is_holder(node) -> bool:
    perceptron = getattr(node, "perceptron", None)
    return perceptron is not None and not getattr(perceptron, "is_encoding_layer", False)


def _shapes(node: ComputeOpMapper) -> Tuple:
    shapes = node.input_shapes
    return tuple(shapes) if shapes is not None and len(shapes) == len(node.sources) else (None,) * len(node.sources)


def plan_lanes(model_repr, declaration: LaneDeclaration) -> LanePlan:
    """The lane plan of ``model_repr`` under ``declaration`` (an add or pool matched with the traced input shapes
    its mapper records; fusion checks them again on the IR)."""
    order, deps, _ = model_repr._ensure_exec_graph()
    readers = model_repr.consumer_map()
    plan = LanePlan(declaration)
    on_chip: set = set()
    open_lane: Dict[int, PerceptronLane] = {}  # id(lane end) -> its lane, while the lane may absorb more

    def only_reader(node, reader) -> bool:
        found = readers.get(id(node), [])
        return len(found) == 1 and found[0] is reader

    def reads_on_chip(holder) -> bool:
        """Every value ``holder`` reads is on chip (it is no segment entry): its sources precede it in ``order``."""
        sources = [s for s in deps.get(holder, []) if s is not None]
        return bool(sources) and all(id(s) in on_chip for s in sources)

    for node in order:
        sources = [s for s in deps.get(node, []) if s is not None]
        if is_holder(node):
            lane = PerceptronLane(node)
            plan.perceptron_lanes.append(lane)
            open_lane[id(node)] = lane
            on_chip.add(id(node))
            continue
        if isinstance(node, WIRING_MAPPERS):
            if sources and all(id(s) in on_chip for s in sources):
                on_chip.add(id(node))
            continue
        if not isinstance(node, ComputeOpMapper):
            continue
        module = node.module
        if (declaration.any and len(sources) == 1 and id(sources[0]) in on_chip
                and zero_pad_form(module, _shapes(node)[0]) is not None):
            plan.pads.append(node)  # wiring: each padded element an off source, which adds 0
            on_chip.add(id(node))
            continue
        if declaration.declares(ADD) and len(sources) == 2 and is_add_form(module, _shapes(node)):
            for index in (0, 1):
                lane = open_lane.get(id(sources[index]))
                operand = sources[1 - index]
                if (lane is not None and lane.holder is sources[index] and lane.add is None and lane.clamp is None
                        and only_reader(sources[index], node) and id(operand) in on_chip
                        and reads_on_chip(sources[index])):
                    lane.add, lane.operand_index = node, 1 - index
                    del open_lane[id(sources[index])]
                    open_lane[id(node)] = lane
                    on_chip.add(id(node))
                    break
            else:
                plan.host_adds.extend(_host_adds(node, sources, open_lane, on_chip, only_reader, reads_on_chip))
            continue
        if declaration.declares(CLAMP) and len(sources) == 1 and is_clamp_form(module):
            lane = open_lane.get(id(sources[0]))
            if lane is not None and lane.clamp is None and only_reader(sources[0], node):
                lane.clamp = node
                del open_lane[id(sources[0])]
                on_chip.add(id(node))
            continue
        shape = _shapes(node)[0] if len(sources) == 1 else None
        if (declaration.declares(POOL) and len(sources) == 1 and id(sources[0]) in on_chip
                and (pool_form(module, shape) is not None if shape is not None else is_pool_form(module))):
            lane = open_lane.get(id(sources[0]))
            if (shape is not None and lane is not None and lane.holder is sources[0] and lane.add is None
                    and lane.reduce is None and only_reader(sources[0], node)):
                lane.reduce = node  # producer-side: the pool sums the holder's accumulators (one rounding)
                del open_lane[id(sources[0])]
                open_lane[id(node)] = lane
            else:
                plan.pool_lanes.append(PoolLane(node, unsigned=_unsigned_value(sources[0], plan)))
            on_chip.add(id(node))
    return plan


def _host_adds(node, sources, open_lane, on_chip, only_reader, reads_on_chip) -> List[HostAdd]:
    """The add ``node`` stays on the host: each input it alone reads whose lane is still open (the holder the add's
    lane would end), with why the add does not join that lane."""
    found = []
    for index in (0, 1):
        holder, operand = sources[index], sources[1 - index]
        lane = open_lane.get(id(holder))
        if (lane is None or lane.holder is not holder or lane.add is not None or lane.clamp is not None
                or not only_reader(holder, node)):
            continue
        reasons = [] if id(operand) in on_chip else [
            f"its operand {getattr(operand, 'name', None) or type(operand).__name__!r} is a host value"]
        if not reads_on_chip(holder):
            reasons.append(f"its holder {getattr(holder, 'name', None)!r} reads host values (a segment entry)")
        found.append(HostAdd(node, holder, ", and ".join(reasons)))
    return found


def _unsigned_value(node, plan: LanePlan) -> bool:
    """Whether an on-chip value is non-negative by construction (a clamp lane's end, or a pool over one, or wiring,
    a zero pad included, over such values)."""
    for lane in plan.perceptron_lanes:
        if lane.end is node:
            return lane.unsigned
    for lane in plan.pool_lanes:
        if lane.pool is node:
            return lane.unsigned
    wiring = isinstance(node, WIRING_MAPPERS) or any(node is pad for pad in plan.pads)
    sources = node.get_source_mappers() if wiring else []
    return bool(sources) and all(_unsigned_value(s, plan) for s in sources)
