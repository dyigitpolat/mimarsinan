"""Shared helpers for layout/HCM placement characterization and parity tests.

Builds a small matrix of representative ``IRGraph`` configs (dense, multi-segment,
fusion, coalescing+scheduled, neuron-splitting, pruned) and a deterministic,
JSON-serialisable *signature* of the resulting ``HybridHardCoreMapping``.

The signature captures exactly the placement decisions the layout single-source
refactor must preserve bit-for-bit: stage order, per-hardcore dimensions, fused
component axons, the full ``soft_core_placements_per_hard_core`` provenance dicts,
and ``neuron_mapping``.  Weights/scales are intentionally excluded -- those are
materialisation and are covered by the separate simulation golden.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore


# ── IR graph builders ──────────────────────────────────────────────────────

def _dense_two_core() -> IRGraph:
    """4 inputs -> core0 (hidden=4) -> core1 (out=2); single neural segment."""
    w1 = np.ones((5, 4), dtype=np.float32) * 0.1
    s1 = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="h", input_sources=s1, core_matrix=w1, latency=0)

    w2 = np.ones((5, 2), dtype=np.float32) * 0.1
    s2 = np.array([IRSource(0, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=1, name="o", input_sources=s2, core_matrix=w2, latency=1)

    out = np.array([IRSource(1, 0), IRSource(1, 1)], dtype=object)
    return IRGraph(nodes=[c1, c2], output_sources=out)


def _multi_segment() -> IRGraph:
    """core0 -> identity ComputeOp -> core1; two neural segments split by host op."""
    w1 = np.ones((3, 2), dtype=np.float32)
    s1 = np.array([IRSource(-2, 0), IRSource(-2, 1), IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="c1", input_sources=s1, core_matrix=w1, latency=0)

    op_src = np.array([IRSource(0, 0), IRSource(0, 1)], dtype=object)
    op = ComputeOp(id=1, name="flat", input_sources=op_src,
                   op_type="identity", input_shape=(2,), output_shape=(2,))

    w2 = np.ones((3, 2), dtype=np.float32)
    s2 = np.array([IRSource(1, 0), IRSource(1, 1), IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=2, name="c2", input_sources=s2, core_matrix=w2, latency=2)

    out = np.array([IRSource(2, 0), IRSource(2, 1)], dtype=object)
    return IRGraph(nodes=[c1, op, c2], output_sources=out)


def _fusion_wide_axon() -> IRGraph:
    """One core with 100 weight axons + bias, 32 neurons -> fuse 2x64 cores."""
    w = np.ones((101, 32), dtype=np.float32) * 0.1
    s = np.array([IRSource(-2, i) for i in range(100)] + [IRSource(-3, 0)], dtype=object)
    c = NeuralCore(id=0, name="large", input_sources=s, core_matrix=w, latency=0)
    out = np.array([IRSource(0, i) for i in range(32)], dtype=object)
    return IRGraph(nodes=[c], output_sources=out)


def _coalescing_wide() -> IRGraph:
    """40 weight axons, 8 neurons -> axon coalescing across cores.

    40 exceeds every core type's max_axons (16, 32) so the wide core is split
    into coalescing fragments before packing.
    """
    rng = np.random.default_rng(42)
    w = rng.uniform(-0.5, 0.5, (40, 8)).astype(np.float32)
    s = np.array([IRSource(-2, i) for i in range(40)], dtype=object)
    c = NeuralCore(id=0, name="wide", input_sources=s, core_matrix=w, latency=0)
    out = np.array([IRSource(0, i) for i in range(8)], dtype=object)
    return IRGraph(nodes=[c], output_sources=out)


def _neuron_split() -> IRGraph:
    """4 axons + bias, 40 neurons on max_neurons=16 -> neuron splitting."""
    rng = np.random.default_rng(7)
    w = rng.uniform(-0.5, 0.5, (5, 40)).astype(np.float32)
    s = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c = NeuralCore(id=0, name="tall", input_sources=s, core_matrix=w, latency=0)
    out = np.array([IRSource(0, i) for i in range(40)], dtype=object)
    return IRGraph(nodes=[c], output_sources=out)


def _pruned() -> IRGraph:
    """Dense two-core graph with a pruned row+col on core0 -> compaction+reindex."""
    ir = _dense_two_core()
    core0 = ir.nodes[0]
    core0.pruned_row_mask = [False, True, False, False, False]
    core0.pruned_col_mask = [False, False, True, False]
    return ir


# name -> (ir_builder, build_kwargs)
CONFIGS: Dict[str, Tuple[Callable[[], IRGraph], Dict[str, Any]]] = {
    "dense_two_core": (
        _dense_two_core,
        {"cores_config": [{"max_axons": 32, "max_neurons": 32, "count": 10}]},
    ),
    "multi_segment": (
        _multi_segment,
        {"cores_config": [{"max_axons": 32, "max_neurons": 32, "count": 10}]},
    ),
    "fusion_wide_axon": (
        _fusion_wide_axon,
        {
            "cores_config": [{"max_axons": 64, "max_neurons": 32, "count": 4}],
            "allow_coalescing": True,
        },
    ),
    "scheduled_wide": (
        _coalescing_wide,
        {
            "cores_config": [
                {"max_axons": 16, "max_neurons": 16, "count": 1},
                {"max_axons": 32, "max_neurons": 16, "count": 2},
            ],
            "allow_scheduling": True,
            "allow_coalescing": True,
        },
    ),
    "neuron_split": (
        _neuron_split,
        {
            "cores_config": [{"max_axons": 32, "max_neurons": 16, "count": 10}],
            "allow_neuron_splitting": True,
        },
    ),
    "pruned": (
        _pruned,
        {"cores_config": [{"max_axons": 32, "max_neurons": 32, "count": 10}]},
    ),
}


# ── Signature extraction ───────────────────────────────────────────────────

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _hardcore_signature(hc: Any) -> Dict[str, Any]:
    return {
        "axons_per_core": int(hc.axons_per_core),
        "neurons_per_core": int(hc.neurons_per_core),
        "available_axons": int(getattr(hc, "available_axons", 0)),
        "available_neurons": int(getattr(hc, "available_neurons", 0)),
        "fused_component_axons": _jsonable(getattr(hc, "fused_component_axons", None)),
    }


def _neuron_mapping_signature(nm: Dict[Any, Any]) -> List[List[int]]:
    items = []
    for (sc_id, neuron), (core_idx, target) in nm.items():
        items.append([int(sc_id), int(neuron), int(core_idx), int(target)])
    items.sort()
    return items


def build_signature(hybrid_mapping: Any) -> Dict[str, Any]:
    """Deterministic, JSON-serialisable signature of placement decisions."""
    stages_sig: List[Dict[str, Any]] = []
    for stage in hybrid_mapping.stages:
        sig: Dict[str, Any] = {
            "kind": stage.kind,
            "schedule_segment_index": stage.schedule_segment_index,
            "schedule_pass_index": stage.schedule_pass_index,
            "input_map": [[s.node_id, s.offset, s.size] for s in stage.input_map],
            "output_map": [[s.node_id, s.offset, s.size] for s in stage.output_map],
        }
        if stage.kind == "compute":
            sig["compute_op_type"] = (
                stage.compute_op.op_type if stage.compute_op else None
            )
        else:
            hcm = stage.hard_core_mapping
            sig["hardcores"] = [_hardcore_signature(hc) for hc in hcm.cores]
            sig["placements"] = _jsonable(hcm.soft_core_placements_per_hard_core)
            sig["neuron_mapping"] = _neuron_mapping_signature(hcm.neuron_mapping)
        stages_sig.append(sig)

    return {
        "stage_kinds": [s.kind for s in hybrid_mapping.stages],
        "num_neural_segments": len(hybrid_mapping.get_neural_segments()),
        "num_compute_ops": len(hybrid_mapping.get_compute_ops()),
        "output_sources_len": int(len(hybrid_mapping.output_sources)),
        "stages": stages_sig,
    }


_PERMISSION_KEYS = (
    "allow_coalescing",
    "allow_neuron_splitting",
    "allow_scheduling",
)


def build_hybrid_for_config(name: str) -> Any:
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy

    builder, kwargs = CONFIGS[name]
    kwargs = dict(kwargs)
    permissions = {k: kwargs.pop(k) for k in _PERMISSION_KEYS if k in kwargs}
    return build_hybrid_hard_core_mapping(
        ir_graph=builder(),
        strategy=MappingStrategy.from_permissions(**permissions),
        **kwargs,
    )


def signature_for_config(name: str) -> Dict[str, Any]:
    return build_signature(build_hybrid_for_config(name))
