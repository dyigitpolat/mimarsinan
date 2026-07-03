"""GUI snapshot module."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.common.best_effort import best_effort

RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"

LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"


def make_resource_ref(source_step_name: str | None, kind: str, rid: str) -> dict:
    ref: dict = {"kind": kind, "rid": rid}
    if source_step_name is not None:
        ref["step"] = source_step_name
    return ref

def _extract_core_connectivity(hcm: Any, segment_index: int) -> list[dict]:
    """Extract inter-core connectivity spans from axon_sources."""
    spans_out: list[dict] = []
    for ci, core in enumerate(hcm.cores):
        axon_spans = None
        with best_effort(f"get axon source spans for core {ci}", logger=logger):
            axon_spans = core.get_axon_source_spans()
        if axon_spans is None:
            continue
        for sp in axon_spans:
            if sp.kind == "core":
                spans_out.append({
                    "src_core": sp.src_core, "src_start": sp.src_start,
                    "src_end": sp.src_end,
                    "dst_core": ci, "dst_start": sp.dst_start,
                    "dst_end": sp.dst_end,
                    "length": sp.length, "kind": "core",
                    "segment": segment_index,
                })
            elif sp.kind == "input":
                spans_out.append({
                    "src_core": -2, "src_start": sp.src_start,
                    "src_end": sp.src_end,
                    "dst_core": ci, "dst_start": sp.dst_start,
                    "dst_end": sp.dst_end,
                    "length": sp.length, "kind": "input",
                    "segment": segment_index,
                })

    with best_effort(f"extract output-source connectivity spans for segment {segment_index}", logger=logger):
        from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
        if hasattr(hcm, "output_sources") and hcm.output_sources is not None:
            out_srcs = hcm.output_sources.flatten().tolist()
            if out_srcs:
                out_spans = compress_spike_sources(out_srcs)
                for sp in out_spans:
                    if sp.kind == "core":
                        spans_out.append({
                            "src_core": sp.src_core, "src_start": sp.src_start,
                            "src_end": sp.src_end,
                            "dst_core": -3, "dst_start": sp.dst_start,
                            "dst_end": sp.dst_end,
                            "length": sp.length, "kind": "output",
                            "segment": segment_index,
                        })

    return spans_out


def _group_consecutive_compute_stages(stages: list[dict]) -> list[dict]:
    """Merge runs of 2+ consecutive compute stages into compute_group entries."""
    result: list[dict] = []
    run: list[dict] = []

    def flush_run() -> None:
        if len(run) == 1:
            result.append(run[0])
        elif len(run) >= 2:
            op_types = [s.get("op_type", "?") for s in run]
            result.append({
                "index": run[0]["index"],
                "kind": "compute_group",
                "name": f"Compute Group ({len(run)} ops)",
                "ops": [
                    {
                        "op_type": s.get("op_type", "?"),
                        "op_name": s.get("op_name", s.get("name", "?")),
                        "input_shape": s.get("input_shape"),
                        "output_shape": s.get("output_shape"),
                    }
                    for s in run
                ],
                "num_ops": len(run),
                "op_types": list(dict.fromkeys(op_types)),
                "is_barrier": True,
            })
        run.clear()

    for stage in stages:
        if stage["kind"] == "compute":
            run.append(stage)
        else:
            flush_run()
            result.append(stage)
    flush_run()
    return result


def _make_segment_spans_extractor(hcm: Any, segment_index: int):
    """Return a memoised zero-arg closure that yields all spans of a segment."""
    state: dict[str, Any] = {"spans": None}
    lock = threading.Lock()

    def get_all() -> list[dict]:
        cached = state["spans"]
        if cached is not None:
            return cached
        with lock:
            if state["spans"] is not None:
                return state["spans"]
            spans: list[dict] = []
            with best_effort(f"lazy connectivity extraction for segment {segment_index}", logger=logger):
                spans = _extract_core_connectivity(hcm, segment_index)
            state["spans"] = spans
            return spans

    return get_all


def _make_per_core_connectivity_producer(get_all_spans, core_index: int):
    """Per-(segment, core) closure returning spans touching *core_index*."""
    def produce() -> list[dict]:
        spans = get_all_spans()
        return [
            sp for sp in spans
            if sp.get("src_core") == core_index or sp.get("dst_core") == core_index
        ]
    return produce


