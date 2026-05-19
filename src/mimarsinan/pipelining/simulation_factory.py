"""Shared construction of hybrid mappings and HCM spiking flows for pipeline steps."""

from __future__ import annotations

from typing import Any

import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


def build_hybrid_mapping_for_pipeline(
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
) -> Any:
    return build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=platform_constraints["cores"],
        allow_neuron_splitting=bool(platform_constraints.get("allow_neuron_splitting", False)),
        allow_scheduling=bool(platform_constraints.get("allow_scheduling", False)),
        allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
    )


def build_spiking_hybrid_flow(
    pipeline,
    hybrid_mapping,
    *,
    preprocessor=None,
) -> SpikingHybridCoreFlow:
    cfg = pipeline.config
    flow = SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid_mapping,
        int(cfg["simulation_steps"]),
        preprocessor,
        cfg["firing_mode"],
        cfg["spike_generation_mode"],
        cfg["thresholding_mode"],
        spiking_mode=cfg.get("spiking_mode", "lif"),
    )
    return flow.to(cfg["device"])


def run_hcm_spiking_test(pipeline, flow: SpikingHybridCoreFlow) -> float:
    """Run soft-core / HCM metric test with subsample or batch limit from config."""
    trainer = BasicTrainer(
        flow,
        pipeline.config["device"],
        DataLoaderFactory(pipeline.data_provider_factory),
        None,
    )
    max_samples = int(pipeline.config.get("max_simulation_samples", 0) or 0)
    sim_batches = pipeline.config.get("simulation_batch_count", None)
    try:
        if max_samples > 0:
            return float(
                trainer.test_on_subsample(
                    max_samples=max_samples,
                    seed=int(pipeline.config.get("seed", 0)),
                )
            )
        return float(trainer.test(max_batches=sim_batches))
    finally:
        trainer.close()
