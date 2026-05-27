"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import (
    _detect_neural_core_liveness,
    _make_bias_strip_producer,
    _make_heatmap_producer,
)

# Bump cautiously: frontend URL builders hard-code these.
RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"


# Per-NeuralCore liveness tags surfaced in the GUI (must match
# ``mimarsinan.mapping.pruning.ir_liveness.NodeLiveness`` for current runs).
LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"  # only for old pickles still containing (1,1) placeholders

def snapshot_ir_graph(
    ir_graph: Any,
    *,
    source_step_name: str | None = None,
) -> tuple[dict, list[ResourceDescriptor]]:
    """Extract IR topology, core stats, and lazy heatmap resource descriptors.

    Parameters
    ----------
    ir_graph
        The IR graph to summarize.
    source_step_name
        Optional name of the step whose resource folder owns the IR-level
        PNG resources (heatmaps, pre-pruning, weight banks). When provided:

        * Every advertised resource ref carries a ``"step": source_step_name``
          field so the frontend can resolve URLs against that step instead
          of the currently-rendering tab's step.
        * The returned descriptor list is empty: the source step has already
          registered (or will register) its own producers, and re-issuing
          them duplicates expensive matplotlib renders. This duplication
          historically caused the snapshot executor's ``wait_idle`` budget
          to expire before all PNGs flushed, leaving missing-image icons
          in the Hardware tab.

        When ``None`` (the default), descriptors are registered as usual
        and refs are emitted without a step name.
    """
    from mimarsinan.mapping.ir import NeuralCore, ComputeOp

    nodes_info: list[dict] = []
    edges: list[dict] = []
    descriptors: list[ResourceDescriptor] = []
    register_descriptors = source_step_name is None

    def _ref(kind: str, rid: str) -> dict:
        ref: dict = {"kind": kind, "rid": rid}
        if source_step_name is not None:
            ref["step"] = source_step_name
        return ref

    neural_cores = []
    compute_ops = []

    # One heatmap resource per weight bank (backend-rendered on first request,
    # not embedded in the summary JSON).
    weight_banks: dict = {}
    try:
        for bank_id, bank in getattr(ir_graph, "weight_banks", {}).items():
            try:
                rid = f"bank/{int(bank_id)}"
                weight_banks[int(bank_id)] = {
                    "has_heatmap": True,
                    "heatmap_resource": _ref(RESOURCE_KIND_IR_BANK_HEATMAP, rid),
                }
                if register_descriptors:
                    descriptors.append(ResourceDescriptor(
                        kind=RESOURCE_KIND_IR_BANK_HEATMAP,
                        rid=rid,
                        producer=_make_heatmap_producer(bank.core_matrix, copy=False),
                        media_type="image/png",
                    ))
            except Exception:
                logger.debug("Failed to register heatmap for weight bank %s", bank_id, exc_info=True)
    except Exception:
        pass

    for topo_idx, node in enumerate(ir_graph.nodes):
        info: dict = {
            "id": node.id,
            "name": node.name,
            "type": "neural_core" if isinstance(node, NeuralCore) else "compute_op",
            "topo_order": topo_idx,
        }

        if isinstance(node, NeuralCore):
            mat = node.get_core_matrix(ir_graph)
            base_group = layer_key_from_node_name(node.name)
            if node.latency is not None:
                group_key = f"{base_group} (L{node.latency})"
            else:
                group_key = base_group

            info["layer_group"] = group_key
            info["axons"] = int(mat.shape[0])
            info["neurons"] = int(mat.shape[1])
            info["threshold"] = float(node.threshold)
            info["activation_scale"] = _t(node.activation_scale)
            info["parameter_scale"] = _t(node.parameter_scale)
            info["input_activation_scale"] = _t(node.input_activation_scale)
            info["latency"] = node.latency
            info["normalization_type"] = getattr(node, "normalization_type", None)
            info["activation_type"] = getattr(node, "activation_type", None)
            info["weight_stats"] = {
                "mean": float(np.mean(mat)),
                "std": float(np.std(mat)),
                "min": float(np.min(mat)),
                "max": float(np.max(mat)),
                "sparsity": float(np.mean(np.abs(mat) < 1e-8)),
                "histogram": _histogram(mat),
            }
            info["psum_group_id"] = node.psum_group_id
            info["psum_role"] = node.psum_role
            info["coalescing_group_id"] = node.coalescing_group_id
            info["coalescing_role"] = node.coalescing_role
            if node.weight_bank_id is not None:
                info["weight_bank_id"] = int(node.weight_bank_id)

            liveness = _detect_neural_core_liveness(node, mat)
            info["liveness"] = liveness

            bias_arr = getattr(node, "hardware_bias", None)
            if bias_arr is not None:
                try:
                    bias_np = np.asarray(bias_arr, dtype=np.float64)
                    if bias_np.size:
                        info["hardware_bias_stats"] = {
                            "size": int(bias_np.size),
                            "min": float(np.min(bias_np)),
                            "max": float(np.max(bias_np)),
                            "abs_max": float(np.max(np.abs(bias_np))),
                            "nonzero": int(np.count_nonzero(bias_np)),
                        }
                except Exception:
                    pass

            # Register a heatmap resource for every NeuralCore (owned and bank-backed).
            core_rid = f"core/{int(node.id)}"
            info["has_heatmap"] = True
            info["heatmap_resource"] = _ref(RESOURCE_KIND_IR_CORE_HEATMAP, core_rid)
            if register_descriptors:
                descriptors.append(ResourceDescriptor(
                    kind=RESOURCE_KIND_IR_CORE_HEATMAP,
                    rid=core_rid,
                    producer=_make_heatmap_producer(mat, copy=False),
                    media_type="image/png",
                ))

            # BIAS_ONLY cores fire from ``hardware_bias`` alone; advertise a
            # tiny bias-strip resource so the heatmap card can show the
            # actual driver alongside the empty weight matrix.
            if (
                liveness == LIVENESS_BIAS_ONLY
                and bias_arr is not None
                and np.asarray(bias_arr).size
            ):
                info["has_bias_resource"] = True
                info["bias_resource"] = _ref(RESOURCE_KIND_IR_CORE_BIAS, core_rid)
                if register_descriptors:
                    descriptors.append(ResourceDescriptor(
                        kind=RESOURCE_KIND_IR_CORE_BIAS,
                        rid=core_rid,
                        producer=_make_bias_strip_producer(bias_arr),
                        media_type="image/png",
                    ))

            pre = getattr(node, "pre_pruning_heatmap", None)
            # Use pre-compaction masks for red markings when present
            # (non–bank-backed cores after ir_pruning).
            row_mask = getattr(node, "pre_pruning_row_mask", None) or getattr(node, "pruned_row_mask", None)
            col_mask = getattr(node, "pre_pruning_col_mask", None) or getattr(node, "pruned_col_mask", None)
            if pre is not None and row_mask is not None and col_mask is not None:
                try:
                    pre_arr = np.array(pre, dtype=np.float64)
                    if pre_arr.shape[0] == len(row_mask) and pre_arr.shape[1] == len(col_mask):
                        info["has_pre_pruning"] = True
                        info["pre_pruning_resource"] = _ref(
                            RESOURCE_KIND_IR_CORE_PRE_PRUNING, core_rid,
                        )
                        info["pre_pruning_axons"] = int(pre_arr.shape[0])
                        info["pre_pruning_neurons"] = int(pre_arr.shape[1])
                        if register_descriptors:
                            descriptors.append(ResourceDescriptor(
                                kind=RESOURCE_KIND_IR_CORE_PRE_PRUNING,
                                rid=core_rid,
                                producer=_make_heatmap_producer(
                                    pre_arr,
                                    pruned_row_mask=list(row_mask),
                                    pruned_col_mask=list(col_mask),
                                    copy=False,
                                ),
                                media_type="image/png",
                            ))
                except Exception:
                    logger.debug("Failed to register pre-pruning heatmap for core %s", node.id, exc_info=True)
            neural_cores.append(info)
        else:
            info["layer_group"] = node.name
            info["op_type"] = node.op_type
            info["input_shape"] = list(node.input_shape) if node.input_shape else None
            info["output_shape"] = list(node.output_shape) if node.output_shape else None
            try:
                info["latency"] = node.latency
            except Exception:
                pass
            try:
                params = node.params
                if params is not None:
                    info["params"] = str(params)[:200]
            except Exception:
                pass
            compute_ops.append(info)

        nodes_info.append(info)

        for src in node.input_sources.flatten():
            if src.node_id >= 0:
                edges.append({"from": int(src.node_id), "to": int(node.id)})
            elif src.node_id == -2:
                edges.append({"from": "input", "to": int(node.id)})
            elif src.node_id == -3:
                edges.append({"from": "const1", "to": int(node.id)})

    try:
        for src in ir_graph.output_sources.flatten():
            if src.node_id >= 0:
                edges.append({"from": int(src.node_id), "to": "output"})
    except Exception:
        logger.debug("Failed to extract output_sources from ir_graph", exc_info=True)

    deduped_edges = _deduplicate_edges(edges)

    node_group_map: dict = {n["id"]: n.get("layer_group", n["name"]) for n in nodes_info}
    node_group_map["input"] = "input"
    node_group_map["const1"] = "const1"
    node_group_map["output"] = "output"

    groups, group_edges = _build_layer_groups(nodes_info, deduped_edges, node_group_map)

    thresholds = [c["threshold"] for c in neural_cores]
    latencies = [c["latency"] for c in neural_cores if c["latency"] is not None]
    axon_counts = [c["axons"] for c in neural_cores]
    neuron_counts = [c["neurons"] for c in neural_cores]

    summary = {
        "num_nodes": len(ir_graph.nodes),
        "num_neural_cores": len(neural_cores),
        "num_compute_ops": len(compute_ops),
        "max_latency": max(latencies) if latencies else 0,
        "nodes": nodes_info,
        "edges": deduped_edges,
        "groups": groups,
        "group_edges": group_edges,
        "weight_banks": weight_banks,
        "threshold_distribution": _histogram(np.array(thresholds)) if thresholds else None,
        "latency_distribution": _histogram(np.array(latencies)) if latencies else None,
        "axon_counts": axon_counts,
        "neuron_counts": neuron_counts,
        "compute_op_types": list({c["op_type"] for c in compute_ops}),
    }
    return summary, descriptors


def _merge_consecutive_compute_groups(groups: list[dict]) -> list[dict]:
    """Merge runs of 2+ consecutive compute-only groups."""
    result: list[dict] = []
    run: list[dict] = []

    def flush_run() -> None:
        if len(run) == 1:
            result.append(run[0])
        elif len(run) >= 2:
            all_node_ids = []
            all_op_types: list[str] = []
            sub_keys: list[str] = []
            for g in run:
                all_node_ids.extend(g.get("node_ids", []))
                all_op_types.extend(g.get("op_types", []))
                sub_keys.append(g["key"])
            merged_key = " + ".join(sub_keys)
            result.append({
                "key": merged_key,
                "order": run[0]["order"],
                "type": "compute_group",
                "num_cores": 0,
                "num_ops": sum(g.get("num_ops", 0) for g in run),
                "node_ids": all_node_ids,
                "threshold_range": None,
                "latency_range": None,
                "axon_range": None,
                "neuron_range": None,
                "op_types": list(dict.fromkeys(all_op_types)),
                "sub_keys": sub_keys,
                "live_core_count": 0,
                "bias_only_count": 0,
                "dead_count": 0,
                "all_dead": False,
                "all_dead_or_bias_only": False,
            })
        run.clear()

    for g in groups:
        if g["type"] == "compute":
            run.append(g)
        else:
            flush_run()
            result.append(g)
    flush_run()

    for idx, g in enumerate(result):
        g["order"] = idx
    return result


def _build_layer_groups(
    nodes: list[dict], edges: list[dict], node_group_map: dict
) -> tuple[list[dict], list[dict]]:
    """Pre-compute group summaries and aggregated group-level edges."""
    groups_by_key: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        gk = node_group_map.get(n["id"], n["name"])
        groups_by_key[gk].append(n)

    has_input = any(e["from"] == "input" for e in edges)
    has_const1 = any(e["from"] == "const1" for e in edges)
    has_output = any(e["to"] == "output" for e in edges)

    for vk in ("input", "const1", "output"):
        if vk not in groups_by_key:
            groups_by_key[vk] = []

    group_order: list[str] = []
    group_min_topo: dict[str, float] = {}
    for gk, members in groups_by_key.items():
        if gk in ("input", "const1", "output"):
            continue
        topos = [m.get("topo_order", float("inf")) for m in members]
        group_min_topo[gk] = min(topos) if topos else float("inf")
        if gk not in group_order:
            group_order.append(gk)
    group_order.sort(key=lambda gk: group_min_topo.get(gk, float("inf")))

    if has_input:
        group_order.insert(0, "input")
    if has_const1:
        group_order.insert(1 if has_input else 0, "const1")
    if has_output:
        group_order.append("output")

    groups: list[dict] = []
    for idx, gk in enumerate(group_order):
        members = groups_by_key.get(gk, [])

        if gk in ("input", "const1", "output"):
            groups.append({
                "key": gk,
                "order": idx,
                "type": "virtual",
                "num_cores": 0,
                "num_ops": 0,
                "node_ids": [],
                "threshold_range": None,
                "latency_range": None,
                "axon_range": None,
                "neuron_range": None,
                "op_types": [],
                "live_core_count": 0,
                "bias_only_count": 0,
                "dead_count": 0,
                "all_dead": False,
                "all_dead_or_bias_only": False,
            })
            continue

        cores = [m for m in members if m["type"] == "neural_core"]
        ops = [m for m in members if m["type"] == "compute_op"]
        thresholds = [c["threshold"] for c in cores if "threshold" in c]
        lats = [c["latency"] for c in cores if c.get("latency") is not None]
        axons = [c["axons"] for c in cores if "axons" in c]
        neurons = [c["neurons"] for c in cores if "neurons" in c]

        live_count = sum(
            1 for c in cores if c.get("liveness", LIVENESS_LIVE) == LIVENESS_LIVE
        )
        bias_only_count = sum(
            1 for c in cores if c.get("liveness") == LIVENESS_BIAS_ONLY
        )
        dead_count = sum(
            1 for c in cores if c.get("liveness") == LIVENESS_DEAD_LEGACY
        )
        all_dead = bool(cores) and dead_count == len(cores)
        all_dead_or_bias_only = bool(cores) and (live_count == 0)

        groups.append({
            "key": gk,
            "order": idx,
            "type": "neural" if cores else "compute",
            "num_cores": len(cores),
            "num_ops": len(ops),
            "node_ids": [m["id"] for m in members],
            "threshold_range": [min(thresholds), max(thresholds)] if thresholds else None,
            "latency_range": [min(lats), max(lats)] if lats else None,
            "axon_range": [min(axons), max(axons)] if axons else None,
            "neuron_range": [min(neurons), max(neurons)] if neurons else None,
            "op_types": list({o.get("op_type", "?") for o in ops}) if ops else [],
            "live_core_count": live_count,
            "bias_only_count": bias_only_count,
            "dead_count": dead_count,
            "all_dead": all_dead,
            "all_dead_or_bias_only": all_dead_or_bias_only,
        })

    groups = _merge_consecutive_compute_groups(groups)

    merged_key_map: dict[str, str] = {}
    for g in groups:
        if g["type"] == "compute_group":
            for sub_key in g.get("sub_keys", []):
                merged_key_map[sub_key] = g["key"]

    node_to_group_key = {}
    for n in nodes:
        raw_key = node_group_map.get(n["id"], n["name"])
        node_to_group_key[n["id"]] = merged_key_map.get(raw_key, raw_key)
    node_to_group_key["input"] = "input"
    node_to_group_key["const1"] = "const1"
    node_to_group_key["output"] = "output"

    ge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for e in edges:
        sg = node_to_group_key.get(e["from"])
        tg = node_to_group_key.get(e["to"])
        if sg and tg and sg != tg:
            ge_counts[(sg, tg)] += 1

    group_edges = [
        {"from": sg, "to": tg, "count": cnt}
        for (sg, tg), cnt in ge_counts.items()
    ]
    return groups, group_edges


def _deduplicate_edges(edges: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for e in edges:
        key = (str(e["from"]), str(e["to"]))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _extract_core_connectivity(hcm: Any, segment_index: int) -> list[dict]:
    """Extract inter-core connectivity spans from axon_sources."""
    spans_out: list[dict] = []
    for ci, core in enumerate(hcm.cores):
        try:
            axon_spans = core.get_axon_source_spans()
        except Exception:
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
    
    # Extract output buffer spans
    try:
        from mimarsinan.mapping.spike_source_spans import compress_spike_sources
        if hasattr(hcm, "output_sources") and hcm.output_sources is not None:
            out_srcs = hcm.output_sources.flatten().tolist()
            if out_srcs:
                out_spans = compress_spike_sources(out_srcs)
                for sp in out_spans:
                    if sp.kind == "core":
                        spans_out.append({
                            "src_core": sp.src_core, "src_start": sp.src_start,
                            "src_end": sp.src_end,
                            "dst_core": -3, "dst_start": sp.dst_start,  # -3 represents output buffer
                            "dst_end": sp.dst_end,
                            "length": sp.length, "kind": "output",
                            "segment": segment_index,
                        })
    except Exception:
        pass

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
    import threading
    state: dict[str, Any] = {"spans": None}
    lock = threading.Lock()

    def get_all() -> list[dict]:
        cached = state["spans"]
        if cached is not None:
            return cached
        with lock:
            if state["spans"] is not None:
                return state["spans"]
            try:
                spans = _extract_core_connectivity(hcm, segment_index)
            except Exception:
                logger.debug(
                    "Lazy connectivity extraction failed for segment %d",
                    segment_index,
                    exc_info=True,
                )
                spans = []
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


