"""Orchestrates per-step GUI snapshots; domain logic in sibling modules."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.snapshot.util.helpers import _t, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.gui.resources import ResourceDescriptor

from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    _group_consecutive_compute_stages as _group_consecutive_compute_stages,
)
from mimarsinan.gui.snapshot.ir_graph.ir_graph_topology_groups import (
    _merge_consecutive_compute_groups as _merge_consecutive_compute_groups,
)

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


from mimarsinan.gui.snapshot.model_snapshot import snapshot_model, snapshot_pruning_layers
from mimarsinan.gui.snapshot.ir_graph import snapshot_ir_graph

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
from mimarsinan.gui.snapshot.sanafe_snapshot import (
    _find_ir_graph_promiser,
    snapshot_sanafe_simulation,
)


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
            with best_effort(f"snapshot model from key {key!r}", logger=logger):
                snapshot["model"] = snapshot_model(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["model"] = kind
            promises_model = (
                step is not None
                and short in set(getattr(step, "promises", ()))
            )
            if promises_model:
                with best_effort("compute planned mapping_performance", logger=logger):
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

        elif short == "ir_graph":
            with best_effort(f"snapshot ir_graph from key {key!r}", logger=logger):
                ir_summary, ir_descs = snapshot_ir_graph(cache.get(key))
                snapshot["ir_graph"] = ir_summary
                descriptors.extend(ir_descs)
                if step is not None:
                    snapshot_key_kinds["ir_graph"] = kind

        elif short == "hard_core_mapping":
            with best_effort(f"snapshot hard_core_mapping from key {key!r}", logger=logger):
                hcm_summary, hcm_descs = snapshot_hard_core_mapping(cache.get(key))
                snapshot["hard_core_mapping"] = hcm_summary
                descriptors.extend(hcm_descs)
                if step is not None:
                    snapshot_key_kinds["hard_core_mapping"] = kind
            with best_effort("compute real mapping_performance", logger=logger):
                real_stats = snapshot_mapping_performance_real(cache.get(key))
                if real_stats is not None:
                    snapshot["mapping_performance"] = real_stats
                    snapshot["mapping_performance_mode"] = "real"
                    if step is not None:
                        snapshot_key_kinds["mapping_performance"] = kind

        elif short == "architecture_search_result":
            with best_effort(f"snapshot search_result from key {key!r}", logger=logger):
                snapshot["search_result"] = snapshot_search_result(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["search_result"] = kind

        elif short == "adaptation_manager":
            with best_effort(f"snapshot adaptation_manager from key {key!r}", logger=logger):
                snapshot["adaptation_manager"] = snapshot_adaptation_manager(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["adaptation_manager"] = kind

        elif short == "activation_scales":
            with best_effort(f"snapshot activation_scales from key {key!r}", logger=logger):
                scales = cache.get(key)
                snapshot["activation_scales"] = [_t(s) for s in scales]
                if step is not None:
                    snapshot_key_kinds["activation_scales"] = kind

        elif short == "platform_constraints_resolved":
            with best_effort(f"snapshot platform_constraints from key {key!r}", logger=logger):
                snapshot["platform_constraints"] = _safe_dict(cache.get(key))
                if step is not None:
                    snapshot_key_kinds["platform_constraints"] = kind

        elif short == "sanafe_simulation_results":
            with best_effort(f"snapshot sanafe_simulation from key {key!r}", logger=logger):
                sf_summary, sf_descs = snapshot_sanafe_simulation(cache.get(key))
                snapshot["sanafe_simulation"] = sf_summary
                descriptors.extend(sf_descs)
                if step is not None:
                    snapshot_key_kinds["sanafe_simulation"] = kind

    # Re-rendering the IR-level PNGs here would exceed GUIHandle.wait_snapshots_idle's snapshot-flush budget and drop images, so embed the summary and tag each resource with the ir_graph promiser's step instead.
    if "hard_core_mapping" in snapshot and "ir_graph" not in snapshot:
        ir_source_step = _find_ir_graph_promiser(pipeline)
        for key in cache.keys():
            short = key.split(".", 1)[-1] if "." in key else key
            if short == "ir_graph":
                with best_effort(f"snapshot ir_graph for hardware tab from key {key!r}", logger=logger):
                    ir_summary, ir_descs = snapshot_ir_graph(
                        cache.get(key), source_step_name=ir_source_step,
                    )
                    snapshot["ir_graph"] = ir_summary
                    descriptors.extend(ir_descs)
                    break

    if step_name == "Pruning Adaptation" and "model" in snapshot:
        for key in cache.keys():
            short = key.split(".", 1)[-1] if "." in key else key
            if short in ("model", "fused_model"):
                with best_effort(f"snapshot pruning layers from key {key!r}", logger=logger):
                    model_obj = cache.get(key)
                    pr_summary, pr_descs = snapshot_pruning_layers(model_obj)
                    snapshot["pruning_layers"] = pr_summary
                    descriptors.extend(pr_descs)
                    if step is not None:
                        snapshot_key_kinds["pruning_layers"] = "new"
                break

    cache_keys = [k.split(".", 1)[-1] if "." in k else k for k in cache.keys() if not k.startswith("__")]
    has_rich_data = any(k in snapshot for k in ("model", "ir_graph", "hard_core_mapping", "search_result"))
    if not has_rich_data:
        snapshot["step_summary"] = {
            "step": step_name,
            "cache_entries": ", ".join(sorted(set(cache_keys))) or "none",
        }

    return snapshot, snapshot_key_kinds, descriptors
