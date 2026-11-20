"""[thru:w3] The batch pipeline's quantities per inference (PL/thru/design.md (c) 'Pricer').

A run of a program of B input slots (``batch_pipeline_slots``) serves B inputs, and a batch of K inferences
(``inference_batch_K``) runs ceil(K / B) rounds, a partial last round costing a whole run. So every count a plane
states per run (``PER_RUN``: passes, barriers, steps, core initializations, evaluated cells, events, carried bytes) is
read per inference as that count times ceil(K / B) / K. One rule on both planes, applied only where B is declared:
undeclared, every quantity is the paper version's. B declared without K states no rounds, so every count per run is
withheld by name. A peak (``carry_peak_live_bytes``) and the write law's figures (already per period / K) are not
counts per run and are never scaled.

An input's span is stated beside them, per input and never per round (``input_span``): the passes from its first to
its last, their steps, the barriers between them and the cores they evaluate, the largest of each over the run's
inputs, which the latency prices as ``latency_input_span_s``."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.quantities.spec import QuantityValue

SLOTS_KEY = "batch_pipeline_slots"
BATCH_KEY = "inference_batch_K"
#: The counts a plane states per run of the program: under the batch pipeline a run serves B inputs.
PER_RUN: Tuple[str, ...] = (
    "pass_count", "sync_count", "latency_steps", "segment_cores", "cores_allocated", "macs", "cells_used",
    "neurons_used", "axons_used", "reprogram_passes", "reprogrammed_bytes", "reprogrammed_cores",
    "connectivity_entries", "local_mem_read_bytes", "local_mem_write_bytes", "carried_raster_bytes",
    "carry_out_bytes", "carry_in_bytes", "carry_onchip_bytes", "total_spikes", "boundary_events", "synaptic_events",
    "synaptic_events_busiest_core", "noc_total_packets", "noc_intra_tile_packets", "noc_inter_tile_packets",
    "noc_total_hops", "noc_value_bytes", "lane_ops", "adc_conversions",
)
SPAN_STEPS, SPAN_SYNCS, SPAN_CORES = "input_span_steps", "input_span_syncs", "input_span_cores"
SPAN_KEYS: Tuple[str, ...] = (SPAN_STEPS, SPAN_SYNCS, SPAN_CORES)
NO_BATCH = ("the run declares batch_pipeline_slots (a run serves B inputs) but no inference_batch_K, so the rounds a "
            "batch runs, and with them each count per inference, are not stated")
NO_SPAN_STEPS = "a pass of the input's span states no executed steps on this plane"

#: ``(ceil(K / B) / K, None)``, ``(None, why)`` under B without K, ``(None, None)`` undeclared.
Rounds = Tuple[Optional[Fraction], Optional[str]]


def declared_slots(platform: Mapping[str, Any]) -> Optional[int]:
    """The inputs a run serves (``batch_pipeline_slots``), None undeclared."""
    slots = platform.get(SLOTS_KEY)
    return None if slots is None else int(slots)


def rounds_per_inference(platform: Mapping[str, Any]) -> Rounds:
    """The runs one inference costs: ceil(K / B) / K under a declared B and K."""
    slots = declared_slots(platform)
    if slots is None:
        return None, None
    batch = platform.get(BATCH_KEY)
    if batch is None:
        return None, NO_BATCH
    k = int(batch)
    return Fraction(-(-k // slots), k), None


def per_inference(values: Dict[str, QuantityValue], rounds: Rounds) -> Dict[str, str]:
    """Every count per run in ``values`` read per inference, in place, under the batch pipeline; returns what it
    withholds (each count per run, by name, under B without K). Undeclared, nothing changes."""
    factor, refusal = rounds
    held: Dict[str, str] = {}
    if factor is None and refusal is None:
        return held
    for key in PER_RUN:
        if key not in values:
            continue
        if refusal is not None:
            values.pop(key)
            held[key] = refusal
            continue
        value = values[key]
        values[key] = QuantityValue(float(Fraction(value.value) * factor), value.provenance)
    return held


def input_span(rows: Sequence[Any], steps_of: Callable[[Any], Optional[int]],
               cores_of: Callable[[Any], int]) -> Tuple[Dict[str, QuantityValue], Dict[str, str]]:
    """An input's span over a plane's pass rows (one per (pass, segment), in program order, each stating the input
    slots it serves, ``PassRow.slots``): per input slot its rows from its first to its last, their steps
    (``steps_of``), the barriers between them (each segment's rows less one) and the cores they evaluate
    (``cores_of``); the largest of each over the slots. Empty for rows that state no slots; the steps withheld by
    name where a row of a span states none."""
    slots = sorted({b for row in rows for b in (getattr(row, "slots", None) or ())})
    if not slots:
        return {}, {}
    steps, syncs, cores = [], [], []
    stated = True
    for b in slots:
        index = [i for i, row in enumerate(rows) if b in (getattr(row, "slots", None) or ())]
        span = rows[index[0]:index[-1] + 1]
        executed = [steps_of(row) for row in span]
        stated = stated and all(s is not None for s in executed)
        steps.append(sum(int(s) for s in executed if s is not None))
        syncs.append(len(span) - len({int(row.segment_index) for row in span}))
        cores.append(sum(int(cores_of(row)) for row in span))
    values = {SPAN_SYNCS: QuantityValue(float(max(syncs)), "static"),
              SPAN_CORES: QuantityValue(float(max(cores)), "static")}
    if not stated:
        return values, {SPAN_STEPS: NO_SPAN_STEPS}
    values[SPAN_STEPS] = QuantityValue(float(max(steps)), "static")
    return values, {}


def evaluated_tiles(row: Any) -> int:
    """A pass row's evaluated tiles: its tiles less the dark ones a wavefront head loads and does not evaluate."""
    return len(row.core_cells) - len(getattr(row, "dark", ()) or ())
