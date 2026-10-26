"""[W15] What an executor reads of a built program, one definition per read, shared by the executors and the static
check (``certification/program_check``), so the check judges the elements a run executes and never a twin of them.

- ``slice_deliveries``: per input slice of an executed unit, its kind (a replayed raster, a count re-encode, a cached
  host train, an encode) and the delivery delay the stage loop applies (``replay_carried_slices`` reads it).
- ``bank_read`` and ``executed_weights``: a core's weight block as the HCM and the TTFS executors upload it, the bank
  fast path (one bank placement over the core's whole used extent) or the dense grid.
- ``MEMOS``: the derived values the executors memoize on a program, a core mapping, a hard core or a host op, each
  with the derivation its accessor computes when the memo is empty (a stale memo steers every run that reads it)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir.dynamic_bank import stage_payload_sources
from mimarsinan.mapping.packing.pass_timing import replay_delays
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources

#: A slice's kind as the stage loop delivers it: its producer's carried raster replayed, the count re-encode a reader
#: of window counts takes (a re-timed level unit, the synchronized count executor), a host op's cached
#: cycle-accurate train taken verbatim, or the encode of its decoded value.
REPLAYED, COUNT, CACHED, ENCODED = "replayed", "count", "cached", "encoded"


@dataclass(frozen=True)
class SliceDelivery:
    """One input slice of an executed unit: its index, its node, its kind and the delay the stage loop applies (0
    where it applies none: a count reader reads no time shift)."""

    index: int
    node: int
    kind: str
    delay: int


def reads_counts(stage, *, count_executor: bool = False) -> bool:
    """Whether ``stage`` reads window counts: a re-timed level unit (``is_retimed_level``), or any unit under the
    synchronized count executor."""
    return bool(count_executor or getattr(stage, "is_retimed_level", False))


def slice_deliveries(stage, carried: Collection[int], *, count_executor: bool = False,
                     cached: Collection[int] = ()) -> List[SliceDelivery]:
    """Per input slice of ``stage``: ``carried`` holds the nodes whose producer published a raster in the stage's
    segment scope, ``cached`` the host ops whose boundary left a cycle-accurate train. A count reader takes every slice
    as its count; otherwise a carried slice is its raster, a cached one its train, any other its encode, and every one
    read at its recorded delay (``replay_delays``)."""
    counts = reads_counts(stage, count_executor=count_executor)
    delays = {} if counts else replay_delays(stage)
    out = []
    for k, s in enumerate(stage.input_map):
        node = int(s.node_id)
        kind = COUNT if counts else REPLAYED if node in carried else CACHED if node in cached else ENCODED
        out.append(SliceDelivery(k, node, kind, int(delays.get(k, 0))))
    return out


@dataclass(frozen=True)
class BankRead:
    """The bank fast path of one core: the bank and its ``[axon0, axon1)`` x ``[neuron0, neuron1)`` block."""

    bank_id: int
    axons: Tuple[int, int]
    neurons: Tuple[int, int]


def bank_read(mapping, core_index: int) -> Optional[BankRead]:
    """The bank a core's weights are uploaded from: a core with exactly one placement record that names a weight bank,
    placed at offsets 0 over the core's whole used extent (``used_axons``, ``used_neurons`` at least one); None for
    every other core, whose weights are its dense grid."""
    core = mapping.cores[core_index]
    rows = getattr(mapping, "soft_core_placements_per_hard_core", None) or []
    records = rows[core_index] if core_index < len(rows) else []
    if len(records) != 1:
        return None
    record = records[0]
    bid = record.get("weight_bank_id")
    axons, neurons = int(record.get("axons", 0)), int(record.get("neurons", 0))
    if (bid is None or int(record.get("axon_offset", 0)) != 0 or int(record.get("neuron_offset", 0)) != 0
            or axons != used_axons(core, min_one=True) or neurons != used_neurons(core, min_one=True)):
        return None
    a0, a1 = record.get("bank_axon_range") or (0, axons)
    n0, n1 = record.get("bank_neuron_range") or (0, neurons)
    return BankRead(int(bid), (int(a0), int(a1)), (int(n0), int(n1)))


def bank_block(mapping, read: BankRead) -> Optional[np.ndarray]:
    """The ``(axons, neurons)`` block ``read`` names in the mapping's weight banks, None when the bank is absent."""
    bank = (getattr(mapping, "weight_banks", None) or {}).get(int(read.bank_id))
    if bank is None:
        return None
    return np.asarray(bank)[read.axons[0]:read.axons[1], read.neurons[0]:read.neurons[1]]


def executed_weights(mapping, core_index: int) -> np.ndarray:
    """The ``(used axons, used neurons)`` weight block the HCM uploads for a core (``build_segment_tensors``): its
    bank block on the bank fast path, else its grid (``get_core_matrix``); a missing bank reads the grid, as the
    TTFS executors read it (the HCM refuses it with a KeyError, the check names it), and so does a bank block whose
    shape is not the used extent (a record's bank range shorter than its ``axons``: the check names it,
    ``bank_content``, instead of stopping on a broadcast error; exact unit's verify minor V4)."""
    core = mapping.cores[core_index]
    want = (used_axons(core, min_one=True), used_neurons(core, min_one=True))
    read = bank_read(mapping, core_index)
    block = None if read is None else bank_block(mapping, read)
    if block is not None and block.shape == want:
        return block
    return np.asarray(core.get_core_matrix())[:want[0], :want[1]]


def consumer_counts(program) -> Dict[int, int]:
    """``{node id: reads}`` over every neural stage's input slices, every host op's sources, the program outputs and
    (N2) each neural stage's payload writes (``dynamic_bank.stage_payload_sources``): the state-buffer refcounts the
    stage loop frees buffers by (``_build_consumer_counts`` and the value flow memoize it)."""
    counts: Dict[int, int] = {}

    def bump(node) -> None:
        counts[int(node)] = counts.get(int(node), 0) + 1

    for stage in program.stages:
        if stage.kind == "neural":
            for s in stage.input_map:
                if s.node_id is not None and int(s.node_id) >= 0:
                    bump(s.node_id)
        elif stage.kind == "compute":
            for src in stage.compute_op.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    bump(src.node_id)
    for src in program.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            bump(src.node_id)
    bindings = getattr(program, "dynamic_bank_bindings", None) or {}
    for stage in program.stages if bindings else ():
        for source in stage_payload_sources(stage, bindings):
            bump(source)
    return counts


def op_source_counts_fresh(op) -> Dict[int, int]:
    """``{producer node id: how many of the op's inputs read it}`` (``op_source_counts`` memoizes it)."""
    counts: Dict[int, int] = {}
    for src in op.input_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            counts[int(src.node_id)] = counts.get(int(src.node_id), 0) + 1
    return counts


@dataclass(frozen=True)
class Memo:
    """A value an executor memoizes on a program element: the element, the attribute, the derivation its accessor
    computes when the memo is empty, and the accessor that reads it."""

    element: str
    attribute: str
    fresh: Callable[[Any], Any]
    accessor: str


MEMOS: Tuple[Memo, ...] = (
    Memo("program", "_carried_output_cache", lambda p: carried_outputs_by_stage(p.stages),
         "carry.carried_output_ids"),
    Memo("program", "_consumer_counts_cache", consumer_counts, "HybridStageIOMixin._build_consumer_counts"),
    Memo("core_mapping", "_output_source_spans",
         lambda m: compress_spike_sources(np.asarray(m.output_sources).flatten().tolist()),
         "HardCoreMapping.get_output_source_spans"),
    Memo("hard_core", "_axon_source_spans", lambda c: compress_spike_sources(c.axon_sources),
         "HardCore.get_axon_source_spans"),
    Memo("host_op", "_source_counts_cache", op_source_counts_fresh, "hybrid_execution.op_source_counts"),
)


def stale_memos(element: str, obj: Any) -> List[str]:
    """The memo attributes of ``obj`` (an element of kind ``element``) that hold another value than their derivation;
    an empty memo (None) is recomputed by its accessor, so it is never stale."""
    out = []
    for memo in MEMOS:
        if memo.element != element:
            continue
        held = getattr(obj, memo.attribute, None)
        if held is not None and held != memo.fresh(obj):
            out.append(memo.attribute)
    return out
