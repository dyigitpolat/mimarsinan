"""Pure snapshot extractors for the GUI monitoring system.

Each function accepts a pipeline artifact and returns a JSON-serializable
dictionary suitable for the web frontend.  No side-effects, no model
mutation.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")


def _t(val: Any) -> float:
    """Convert tensor/ndarray/scalar to plain float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def _histogram(arr: np.ndarray, bins: int = 50) -> dict:
    """Return a compact histogram dict (counts, bin_edges)."""
    if arr.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(arr.flatten(), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


# ---------------------------------------------------------------------------
# Model snapshot
# ---------------------------------------------------------------------------

def snapshot_model(model: Any) -> dict:
    """Extract per-layer weight/bias statistics and architecture info."""
    import torch

    layers: list[dict] = []
    total_params = 0

    perceptrons = _get_model_perceptrons(model)

    for idx, p in enumerate(perceptrons):
        layer_info: dict = {"index": idx, "name": getattr(p, "name", f"perceptron_{idx}")}
        try:
            w = p.layer.weight.data.detach().cpu().numpy()
            layer_info["weight"] = {
                "shape": list(w.shape),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "histogram": _histogram(w),
                "sparsity": float(np.mean(np.abs(w) < 1e-8)),
            }
            total_params += w.size
        except Exception:
            layer_info["weight"] = None

        try:
            b = p.layer.bias.data.detach().cpu().numpy()
            layer_info["bias"] = {
                "shape": list(b.shape),
                "mean": float(np.mean(b)),
                "std": float(np.std(b)),
                "min": float(np.min(b)),
                "max": float(np.max(b)),
                "histogram": _histogram(b),
            }
            total_params += b.size
        except Exception:
            layer_info["bias"] = None

        layer_info["activation_scale"] = _safe_scalar(p, "activation_scale")
        layer_info["parameter_scale"] = _safe_scalar(p, "parameter_scale")
        layer_info["input_scale"] = _safe_scalar(p, "input_scale")

        layers.append(layer_info)

    try:
        total_params_torch = sum(p.numel() for p in model.parameters())
    except Exception:
        total_params_torch = total_params

    return {
        "total_params": int(total_params_torch),
        "num_layers": len(layers),
        "layers": layers,
    }


def _get_model_perceptrons(model: Any) -> list:
    """Try multiple strategies to extract layer-like objects from the model."""
    try:
        perceptrons = model.get_perceptrons()
        if perceptrons:
            return perceptrons
    except Exception:
        pass

    try:
        if hasattr(model, "perceptrons"):
            return list(model.perceptrons)
    except Exception:
        pass

    try:
        import torch.nn as nn
        children = list(model.children())
        if children:
            linear_layers = []
            for i, child in enumerate(children):
                if isinstance(child, nn.Linear):
                    wrapper = type("_Wrapper", (), {"layer": child, "name": f"linear_{i}"})()
                    wrapper.activation_scale = None
                    wrapper.parameter_scale = None
                    wrapper.input_scale = None
                    linear_layers.append(wrapper)
            if linear_layers:
                return linear_layers
    except Exception:
        pass

    logger.debug("Could not extract perceptrons/layers from model %s", type(model).__name__)
    return []


def _safe_scalar(obj: Any, attr: str) -> float | None:
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return None
        return _t(val)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IR Graph snapshot
# ---------------------------------------------------------------------------

_RE_CONV_POS = re.compile(r"^(.*)_pos\d+_\d+_g\d+$")
_RE_FC_TILE = re.compile(r"^(.*)_tile_\d+_\d+$")


def _layer_key_from_node_name(name: str) -> str:
    """Best-effort grouping key that collapses per-position/per-tile cores into a layer stack."""
    s = str(name)
    m = _RE_CONV_POS.match(s)
    if m:
        return m.group(1)
    m = _RE_FC_TILE.match(s)
    if m:
        return m.group(1)
    if "_psum_" in s:
        return s.split("_psum_", 1)[0]
    return s


def snapshot_ir_graph(ir_graph: Any) -> dict:
    """Extract topology, core stats, thresholds, latencies from an IRGraph.

    Includes layer-group annotations and pre-computed group summaries for
    the layered topology frontend view.  Special virtual nodes (input,
    const1, output) and their edges are included.

    Nodes receive a ``topo_order`` field (their index in the topologically
    sorted ``ir_graph.nodes``) so that the frontend can interleave ComputeOps
    at their correct position in the data flow.
    """
    from mimarsinan.mapping.ir import NeuralCore, ComputeOp

    nodes_info: list[dict] = []
    edges: list[dict] = []

    neural_cores = []
    compute_ops = []

    for topo_idx, node in enumerate(ir_graph.nodes):
        info: dict = {
            "id": node.id,
            "name": node.name,
            "type": "neural_core" if isinstance(node, NeuralCore) else "compute_op",
            "topo_order": topo_idx,
        }

        if isinstance(node, NeuralCore):
            mat = node.core_matrix
            group_key = _layer_key_from_node_name(node.name)
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
            info["psum_role"] = node.psum_role
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

    return {
        "num_nodes": len(ir_graph.nodes),
        "num_neural_cores": len(neural_cores),
        "num_compute_ops": len(compute_ops),
        "max_latency": max(latencies) if latencies else 0,
        "nodes": nodes_info,
        "edges": deduped_edges,
        "groups": groups,
        "group_edges": group_edges,
        "threshold_distribution": _histogram(np.array(thresholds)) if thresholds else None,
        "latency_distribution": _histogram(np.array(latencies)) if latencies else None,
        "axon_counts": axon_counts,
        "neuron_counts": neuron_counts,
        "compute_op_types": list({c["op_type"] for c in compute_ops}),
    }


def _build_layer_groups(
    nodes: list[dict], edges: list[dict], node_group_map: dict
) -> tuple[list[dict], list[dict]]:
    """Pre-compute group summaries and aggregated group-level edges.

    Handles mixed int/str node IDs (regular nodes use ints, virtual nodes
    like "input"/"const1"/"output" use strings).
    """
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
            })
            continue

        cores = [m for m in members if m["type"] == "neural_core"]
        ops = [m for m in members if m["type"] == "compute_op"]
        thresholds = [c["threshold"] for c in cores if "threshold" in c]
        lats = [c["latency"] for c in cores if c.get("latency") is not None]
        axons = [c["axons"] for c in cores if "axons" in c]
        neurons = [c["neurons"] for c in cores if "neurons" in c]

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
        })

    node_to_group_key = {}
    for n in nodes:
        node_to_group_key[n["id"]] = node_group_map.get(n["id"], n["name"])
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


# ---------------------------------------------------------------------------
# Hardware mapping snapshot
# ---------------------------------------------------------------------------

def _downsample_matrix(mat: np.ndarray, max_size: int = 96) -> list:
    """Downsample a 2D matrix to at most max_size x max_size.

    Uses a max-abs representative in each block so non-zero utilized
    weights remain visible even when large unused padded regions exist.
    """
    h, w = mat.shape
    if h == 0 or w == 0:
        return [[0.0]]
    bh = max(1, int(np.ceil(h / max_size)))
    bw = max(1, int(np.ceil(w / max_size)))
    new_h = (h + bh - 1) // bh
    new_w = (w + bw - 1) // bw
    out = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            block = mat[i * bh:min((i + 1) * bh, h), j * bw:min((j + 1) * bw, w)]
            flat = block.reshape(-1)
            if flat.size == 0:
                out[i, j] = 0.0
            else:
                idx = int(np.argmax(np.abs(flat)))
                out[i, j] = float(flat[idx])
    return out.tolist()


def _extract_core_connectivity(hcm: Any, segment_index: int) -> list[dict]:
    """Extract detailed inter-core connectivity spans from axon_sources.

    Returns a list of span dicts with exact axon/neuron ranges so the
    frontend can highlight the specific side-segments of each core.
    """
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


def snapshot_hard_core_mapping(mapping: Any) -> dict:
    """Extract utilization, packing, stage flow, and per-core detail from HybridHardCoreMapping."""
    stages_info: list[dict] = []
    all_core_utils: list[dict] = []
    neural_segment_idx = 0

    for i, stage in enumerate(mapping.stages):
        stage_info: dict = {"index": i, "kind": stage.kind, "name": stage.name}

        if stage.kind == "neural" and stage.hard_core_mapping is not None:
            hcm = stage.hard_core_mapping
            seg_idx = neural_segment_idx
            stage_info["segment_index"] = seg_idx
            neural_segment_idx += 1

            cores_detail: list[dict] = []
            for ci, core in enumerate(hcm.cores):
                used_axons = core.axons_per_core - core.available_axons
                used_neurons = core.neurons_per_core - core.available_neurons
                total = core.axons_per_core * core.neurons_per_core
                used = used_axons * used_neurons
                utilization = used / total if total > 0 else 0.0
                core_d: dict = {
                    "core_index": ci,
                    "axons_per_core": core.axons_per_core,
                    "neurons_per_core": core.neurons_per_core,
                    "used_axons": used_axons,
                    "used_neurons": used_neurons,
                    "display_axons": used_axons if used_axons > 0 else core.axons_per_core,
                    "display_neurons": used_neurons if used_neurons > 0 else core.neurons_per_core,
                    "utilization": utilization,
                    "threshold": float(core.threshold) if core.threshold is not None else None,
                    "latency": core.latency,
                }
                try:
                    core_d["heatmap"] = _downsample_matrix(core.core_matrix)
                except Exception:
                    pass
                cores_detail.append(core_d)
                all_core_utils.append(core_d)
            stage_info["num_cores"] = len(hcm.cores)
            stage_info["cores"] = cores_detail

            try:
                stage_info["connectivity"] = _extract_core_connectivity(hcm, seg_idx)
            except Exception:
                logger.debug("Failed to extract connectivity for stage %d", i, exc_info=True)

            try:
                stage_info["input_map"] = [
                    {"node_id": s.node_id, "offset": s.offset, "size": s.size}
                    for s in stage.input_map
                ]
                stage_info["output_map"] = [
                    {"node_id": s.node_id, "offset": s.offset, "size": s.size}
                    for s in stage.output_map
                ]
            except Exception:
                logger.debug("Failed to extract io_map for stage %d", i, exc_info=True)

        elif stage.kind == "compute" and stage.compute_op is not None:
            stage_info["op_type"] = stage.compute_op.op_type
            stage_info["op_name"] = stage.compute_op.name
            stage_info["input_shape"] = list(stage.compute_op.input_shape) if getattr(stage.compute_op, "input_shape", None) is not None else None
            stage_info["output_shape"] = list(stage.compute_op.output_shape) if getattr(stage.compute_op, "output_shape", None) is not None else None
            stage_info["is_barrier"] = True

        stages_info.append(stage_info)

    core_reuse = _compute_core_reuse(stages_info)
    global_core_layout = _compute_global_core_layout(stages_info)

    utilizations = [c["utilization"] for c in all_core_utils]
    return {
        "num_stages": len(mapping.stages),
        "num_neural_segments": len(mapping.get_neural_segments()),
        "num_compute_ops": len(mapping.get_compute_ops()),
        "total_cores": len(all_core_utils),
        "stages": stages_info,
        "core_reuse": core_reuse,
        "global_core_layout": global_core_layout,
        "utilization_histogram": _histogram(np.array(utilizations)) if utilizations else None,
        "mean_utilization": float(np.mean(utilizations)) if utilizations else 0.0,
    }


def _compute_global_core_layout(stages: list[dict]) -> list[dict]:
    """Compute the global hardware core layout across all neural segments.

    For each unique core dimension (axons, neurons), returns the maximum
    count seen in any single segment.  This is the minimum hardware
    requirement: the frontend renders every segment with this layout,
    leaving unused slots as placeholders.
    """
    dim_max_count: dict[tuple[int, int], int] = defaultdict(int)
    for s in stages:
        if s["kind"] != "neural" or "cores" not in s:
            continue
        seg_counts: dict[tuple[int, int], int] = defaultdict(int)
        for c in s["cores"]:
            dim = (c["axons_per_core"], c["neurons_per_core"])
            seg_counts[dim] += 1
        for dim, cnt in seg_counts.items():
            dim_max_count[dim] = max(dim_max_count[dim], cnt)
    return [
        {"axons_per_core": a, "neurons_per_core": n, "count": cnt}
        for (a, n), cnt in sorted(dim_max_count.items())
    ]


def _compute_core_reuse(stages: list[dict]) -> dict:
    """Compute per-core-dimension reuse across neural segments.

    Groups cores by (axons_per_core, neurons_per_core) to identify
    hardware core configurations reused across segments.
    """
    dim_to_segments: dict[str, list[int]] = defaultdict(list)
    for s in stages:
        if s["kind"] != "neural" or "cores" not in s:
            continue
        seg_idx = s.get("segment_index", s["index"])
        for c in s["cores"]:
            dim_key = f"{c['axons_per_core']}x{c['neurons_per_core']}"
            if seg_idx not in dim_to_segments[dim_key]:
                dim_to_segments[dim_key].append(seg_idx)
    return {
        "core_configs": [
            {"dimensions": k, "segments": v, "num_segments": len(v)}
            for k, v in dim_to_segments.items()
        ]
    }


# ---------------------------------------------------------------------------
# Architecture search result snapshot
# ---------------------------------------------------------------------------

def snapshot_search_result(result: Any) -> dict:
    """Extract Pareto front, candidates, objectives from a SearchResult.

    Handles both the original SearchResult dataclass and the dict-serialized
    form produced by ``_search_result_to_jsonable()`` in ArchitectureSearchStep.
    """
    if isinstance(result, dict):
        return _snapshot_search_result_dict(result)
    return _snapshot_search_result_obj(result)


def _snapshot_search_result_dict(d: dict) -> dict:
    """Handle the dict form: keys are objectives, best, pareto_front, all_candidates, history.
    Each candidate has configuration, objectives, metadata."""
    best = None
    try:
        b = d["best"]
        best = {
            "config": _safe_dict(b.get("configuration", b.get("config", {}))),
            "objectives": _safe_dict(b.get("objectives", {})),
        }
    except Exception:
        logger.debug("Failed to extract best from dict search result", exc_info=True)

    pareto = []
    try:
        for c in d.get("pareto_front", []):
            pareto.append({
                "config": _safe_dict(c.get("configuration", c.get("config", {}))),
                "objectives": _safe_dict(c.get("objectives", {})),
            })
    except Exception:
        logger.debug("Failed to extract pareto_front from dict search result", exc_info=True)

    history = []
    try:
        for h in d.get("history", []):
            history.append(_safe_dict(h))
    except Exception:
        logger.debug("Failed to extract history from dict search result", exc_info=True)

    objectives = []
    try:
        for obj in d.get("objectives", []):
            if isinstance(obj, dict):
                objectives.append({"name": obj.get("name", "?"), "goal": obj.get("goal", "?")})
            else:
                objectives.append({"name": getattr(obj, "name", "?"), "goal": getattr(obj, "goal", "?")})
    except Exception:
        logger.debug("Failed to extract objectives from dict search result", exc_info=True)

    all_candidates = d.get("all_candidates", [])
    return {
        "best": best,
        "pareto_front": pareto,
        "num_candidates": len(all_candidates),
        "history": history,
        "objectives": objectives,
    }


def _snapshot_search_result_obj(result: Any) -> dict:
    """Handle the original SearchResult dataclass form."""
    try:
        best = {
            "config": _safe_dict(result.best.config if hasattr(result.best, 'config') else result.best.configuration),
            "objectives": _safe_dict(result.best.objectives),
        }
    except Exception:
        best = None

    pareto = []
    try:
        for c in result.pareto_front:
            pareto.append({
                "config": _safe_dict(c.config if hasattr(c, 'config') else c.configuration),
                "objectives": _safe_dict(c.objectives),
            })
    except Exception:
        pass

    history = []
    try:
        for h in result.history:
            history.append(_safe_dict(h))
    except Exception:
        pass

    objectives = []
    try:
        for obj in result.objectives:
            objectives.append({"name": obj.name, "goal": obj.goal})
    except Exception:
        pass

    return {
        "best": best,
        "pareto_front": pareto,
        "num_candidates": len(getattr(result, "all_candidates", [])),
        "history": history,
        "objectives": objectives,
    }


# ---------------------------------------------------------------------------
# Adaptation manager snapshot
# ---------------------------------------------------------------------------

def snapshot_adaptation_manager(manager: Any) -> dict:
    """Extract current adaptation rates."""
    return {
        "clamp_rate": getattr(manager, "clamp_rate", None),
        "shift_rate": getattr(manager, "shift_rate", None),
        "quantization_rate": getattr(manager, "quantization_rate", None),
        "scale_rate": getattr(manager, "scale_rate", None),
        "noise_rate": getattr(manager, "noise_rate", None),
    }


# ---------------------------------------------------------------------------
# Pipeline cache snapshot dispatcher
# ---------------------------------------------------------------------------

def build_step_snapshot(pipeline: Any, step_name: str) -> dict:
    """Build a rich snapshot from the pipeline cache after a step completes.

    Inspects the cache for known entries and extracts visualizable data.
    Safe to call even if entries are missing.
    """
    snapshot: dict = {"step_name": step_name}
    cache = pipeline.cache

    for key in cache.keys():
        short = key.split(".", 1)[-1] if "." in key else key

        if short == "model":
            try:
                snapshot["model"] = snapshot_model(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot model from key %r", key, exc_info=True)

        elif short == "ir_graph":
            try:
                snapshot["ir_graph"] = snapshot_ir_graph(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot ir_graph from key %r", key, exc_info=True)

        elif short == "hard_core_mapping":
            try:
                snapshot["hard_core_mapping"] = snapshot_hard_core_mapping(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot hard_core_mapping from key %r", key, exc_info=True)

        elif short == "architecture_search_result":
            try:
                snapshot["search_result"] = snapshot_search_result(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot search_result from key %r", key, exc_info=True)

        elif short == "adaptation_manager":
            try:
                snapshot["adaptation_manager"] = snapshot_adaptation_manager(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot adaptation_manager from key %r", key, exc_info=True)

        elif short == "activation_scales":
            try:
                scales = cache.get(key)
                snapshot["activation_scales"] = [_t(s) for s in scales]
            except Exception:
                logger.debug("Failed to snapshot activation_scales from key %r", key, exc_info=True)

        elif short == "platform_constraints_resolved":
            try:
                snapshot["platform_constraints"] = _safe_dict(cache.get(key))
            except Exception:
                logger.debug("Failed to snapshot platform_constraints from key %r", key, exc_info=True)

    cache_keys = [k.split(".", 1)[-1] if "." in k else k for k in cache.keys() if not k.startswith("__")]
    has_rich_data = any(k in snapshot for k in ("model", "ir_graph", "hard_core_mapping", "search_result"))
    if not has_rich_data:
        snapshot["step_summary"] = {
            "step": step_name,
            "cache_entries": ", ".join(sorted(set(cache_keys))) or "none",
        }

    return snapshot


def _safe_dict(obj: Any) -> Any:
    """Recursively convert an object to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
