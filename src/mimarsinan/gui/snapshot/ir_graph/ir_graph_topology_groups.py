"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import (
    _detect_neural_core_liveness,
    _make_bias_strip_producer,
    _make_heatmap_producer,
)


from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    LIVENESS_BIAS_ONLY,
    LIVENESS_DEAD_LEGACY,
    LIVENESS_LIVE,
)

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
