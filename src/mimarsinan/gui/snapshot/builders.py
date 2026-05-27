"""Orchestrates per-step GUI snapshots; domain logic in sibling modules."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor

from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import _group_consecutive_compute_stages
from mimarsinan.gui.snapshot.ir_graph.ir_graph_topology_groups import _merge_consecutive_compute_groups

from mimarsinan.gui.snapshot.util.constants import (
    LIVENESS_BIAS_ONLY,
    LIVENESS_DEAD_LEGACY,
    LIVENESS_LIVE,
    RESOURCE_KIND_CONNECTIVITY,
    RESOURCE_KIND_HARD_CORE_HEATMAP,
    RESOURCE_KIND_IR_BANK_HEATMAP,
    RESOURCE_KIND_IR_CORE_BIAS,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
)


from mimarsinan.gui.snapshot.heatmap import _make_heatmap_producer, _make_bias_strip_producer
from mimarsinan.gui.snapshot.model_snapshot import snapshot_model, snapshot_pruning_layers
from mimarsinan.gui.snapshot.ir_graph import snapshot_ir_graph

# Re-export for tests and legacy import paths.
__all__ = [
    "LIVENESS_BIAS_ONLY",
    "LIVENESS_DEAD_LEGACY",
    "LIVENESS_LIVE",
    "RESOURCE_KIND_CONNECTIVITY",
    "RESOURCE_KIND_HARD_CORE_HEATMAP",
    "RESOURCE_KIND_IR_BANK_HEATMAP",
    "RESOURCE_KIND_IR_CORE_BIAS",
    "RESOURCE_KIND_IR_CORE_HEATMAP",
    "RESOURCE_KIND_IR_CORE_PRE_PRUNING",
    "RESOURCE_KIND_PRUNING_LAYER_HEATMAP",
    "build_step_snapshot",
    "snapshot_ir_graph",
]
from mimarsinan.gui.snapshot.mapping_snapshot import (
    snapshot_mapping_performance_planned,
    snapshot_mapping_performance_real,
    snapshot_hard_core_mapping,
)
from mimarsinan.gui.snapshot.search_snapshot import snapshot_search_result
from mimarsinan.gui.snapshot.adaptation_snapshot import snapshot_adaptation_manager
from mimarsinan.gui.snapshot.sanafe_snapshot import snapshot_sanafe_simulation


def build_step_snapshot(
    pipeline: Any,
    step_name: str,
    step: Any = None,
) -> tuple[dict, dict[str, str], list[ResourceDescriptor]]:
    """Build a rich snapshot from the pipeline cache after a step completes."""
    snapshot: dict = {"step_name": step_name}
    snapshot_key_kinds: dict[str, str] = {}
    descriptors: list[ResourceDescriptor] = []
    cache = pipeline.cache

    if step is None:
        for name, s in pipeline.steps:
            if name == step_name:
                step = s
                break

    allowed_cache_keys: set[str] | None = None
    if step is not None:
        allowed_cache_keys = set(getattr(step, "promises", ())) | set(
            getattr(step, "updates", ())
        )

    for key in cache.keys():
        short = key.split(".", 1)[-1] if "." in key else key
        if allowed_cache_keys is not None and short not in allowed_cache_keys:
            continue
        snapshot_key = _CACHE_KEY_TO_SNAPSHOT_KEY.get(short)
        if snapshot_key is None:
            continue

        kind = "new" if (step is not None and short in getattr(step, "promises", ())) else "edited"
        if step is not None and short in getattr(step, "updates", ()):
            kind = "edited"

        if short in ("model", "fused_model"):
            try:
                snapshot["model"] = snapshot_model(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["model"] = kind
            except Exception:
                logger.debug("Failed to snapshot model from key %r", key, exc_info=True)
            promises_model = (
                step is not None
                and short in set(getattr(step, "promises", ()))
            )
            if promises_model:
                try:
                    pcfg = None
                    for k2 in cache.keys():
                        short2 = k2.split(".", 1)[-1] if "." in k2 else k2
                        if short2 == "platform_constraints_resolved":
                            pcfg = _safe_dict(cache.get(k2))
                            break
                    pipeline_cfg = getattr(pipeline, "config", {}) or {}
                    planned = snapshot_mapping_performance_planned(
                        cache.get(key),
                        pcfg,
                        input_shape=pipeline_cfg.get("input_shape"),
                        num_classes=pipeline_cfg.get("num_classes"),
                    )
                    if planned is not None:
                        snapshot["mapping_performance"] = planned
                        snapshot["mapping_performance_mode"] = "planned"
                        snapshot_key_kinds["mapping_performance"] = "new"
                except Exception:
                    logger.debug("Failed to compute planned mapping_performance", exc_info=True)

        elif short == "ir_graph":
            try:
                ir_summary, ir_descs = snapshot_ir_graph(cache.get(key))
                snapshot["ir_graph"] = ir_summary
                descriptors.extend(ir_descs)
                if step is not None:
                    snapshot_key_kinds["ir_graph"] = kind
            except Exception:
                logger.debug("Failed to snapshot ir_graph from key %r", key, exc_info=True)

        elif short == "hard_core_mapping":
            try:
                hcm_summary, hcm_descs = snapshot_hard_core_mapping(cache.get(key))
                snapshot["hard_core_mapping"] = hcm_summary
                descriptors.extend(hcm_descs)
                if step is not None:
                    snapshot_key_kinds["hard_core_mapping"] = kind
            except Exception:
                logger.debug("Failed to snapshot hard_core_mapping from key %r", key, exc_info=True)
            try:
                real_stats = snapshot_mapping_performance_real(cache.get(key))
                if real_stats is not None:
                    snapshot["mapping_performance"] = real_stats
                    snapshot["mapping_performance_mode"] = "real"
                    if step is not None:
                        snapshot_key_kinds["mapping_performance"] = kind
            except Exception:
                logger.debug("Failed to compute real mapping_performance", exc_info=True)

        elif short == "architecture_search_result":
            try:
                snapshot["search_result"] = snapshot_search_result(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["search_result"] = kind
            except Exception:
                logger.debug("Failed to snapshot search_result from key %r", key, exc_info=True)

        elif short == "adaptation_manager":
            try:
                snapshot["adaptation_manager"] = snapshot_adaptation_manager(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["adaptation_manager"] = kind
            except Exception:
                logger.debug("Failed to snapshot adaptation_manager from key %r", key, exc_info=True)

        elif short == "activation_scales":
            try:
                scales = cache.get(key)
                snapshot["activation_scales"] = [_t(s) for s in scales]
                if step is not None:
                    snapshot_key_kinds["activation_scales"] = kind
            except Exception:
                logger.debug("Failed to snapshot activation_scales from key %r", key, exc_info=True)

        elif short == "platform_constraints_resolved":
            try:
                snapshot["platform_constraints"] = _safe_dict(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["platform_constraints"] = kind
            except Exception:
                logger.debug("Failed to snapshot platform_constraints from key %r", key, exc_info=True)

        elif short == "sanafe_simulation_results":
            try:
                sf_summary, sf_descs = snapshot_sanafe_simulation(cache.get(key))
                snapshot["sanafe_simulation"] = sf_summary
                descriptors.extend(sf_descs)
                if step is not None:
                    snapshot_key_kinds["sanafe_simulation"] = kind
            except Exception:
                logger.debug("Failed to snapshot sanafe_simulation from key %r", key, exc_info=True)

    # Hardware tab needs ir_graph to show soft-core detail pane when clicking
    # heatmap regions, but the IR-level PNGs (heatmap, pre-pruning, weight
    # banks) have already been rendered by the step that *promised* ir_graph
    # (typically Soft Core Mapping). Re-rendering them here both wastes ~30+
    # seconds of matplotlib work and previously caused
    # ``GUIHandle.wait_snapshots_idle``'s 30 s budget to expire mid-flush,
    # leaving missing-image icons for cores past the cut-off in the Hardware
    # and IR Graph tabs (only ~300/576 PNGs landed on disk for HCM).
    #
    # Instead we embed the summary and tag every resource ref with
    # ``step = <ir_graph promiser>`` so the frontend resolves URLs against
    # that step's already-persisted resource folder.
    if "hard_core_mapping" in snapshot and "ir_graph" not in snapshot:
        ir_source_step = _find_ir_graph_promiser(pipeline)
        for key in cache.keys():
            short = key.split(".", 1)[-1] if "." in key else key
            if short == "ir_graph":
                try:
                    ir_summary, ir_descs = snapshot_ir_graph(
                        cache.get(key), source_step_name=ir_source_step,
                    )
                    snapshot["ir_graph"] = ir_summary
                    # ir_descs is empty when source_step_name is set; extending
                    # is a no-op in that case but keeps the contract uniform if
                    # the lookup ever fails to find an owning step.
                    descriptors.extend(ir_descs)
                    break
                except Exception:
                    logger.debug("Failed to snapshot ir_graph for hardware tab from key %r", key, exc_info=True)

    if step_name == "Pruning Adaptation" and "model" in snapshot:
        for key in cache.keys():
            short = key.split(".", 1)[-1] if "." in key else key
            if short in ("model", "fused_model"):
                try:
                    model_obj = cache.get(key)
                    pr_summary, pr_descs = snapshot_pruning_layers(model_obj)
                    snapshot["pruning_layers"] = pr_summary
                    descriptors.extend(pr_descs)
                    if step is not None:
                        snapshot_key_kinds["pruning_layers"] = "new"
                except Exception:
                    logger.debug("Failed to snapshot pruning layers from key %r", key, exc_info=True)
                break

    cache_keys = [k.split(".", 1)[-1] if "." in k else k for k in cache.keys() if not k.startswith("__")]
    has_rich_data = any(k in snapshot for k in ("model", "ir_graph", "hard_core_mapping", "search_result"))
    if not has_rich_data:
        snapshot["step_summary"] = {
            "step": step_name,
            "cache_entries": ", ".join(sorted(set(cache_keys))) or "none",
        }

    return snapshot, snapshot_key_kinds, descriptors
