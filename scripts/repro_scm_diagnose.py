#!/usr/bin/env python3
"""Diagnose NF vs HCM on saved run."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mimarsinan.data_handling.data_providers.mnist_data_provider  # noqa: F401
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import _CycleAccurateForward


def accuracy(model, device, pf, max_samples=200):
    trainer = BasicTrainer(model, device, DataLoaderFactory(pf), None)
    trainer.test_loader.num_workers = 0
    return trainer.test_on_subsample(max_samples=max_samples, seed=0)


def main() -> None:
    run_dir = ROOT / "generated/mnist_hard_all_lif_phased_deployment_run_20260520_084932"
    cfg = json.loads((run_dir / "_RUN_CONFIG/config.json").read_text())
    pc_res = json.loads(
        (run_dir / "Model Configuration.platform_constraints_resolved.json").read_text()
    )
    pc = cfg["platform_constraints"]
    dp = cfg["deployment_parameters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf = BasicDataProviderFactory(cfg["data_provider_name"], "")

    obj = torch.load(
        run_dir / "Normalization Fusion.fused_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = obj[0] if isinstance(obj, tuple) else obj
    model.eval().to(device)

    print("=== NF paths ===")
    print(f"current forward: {accuracy(model, device, pf):.4f}")

    if isinstance(model.forward, _CycleAccurateForward):
        wrapper = model.forward
        native = nn.Module()

        def _forward(x):
            return wrapper._unpatched_forward(x)

        native.forward = _forward  # type: ignore[method-assign]
        print(f"native single-step forward: {accuracy(native, device, pf):.4f}")

    ir_graph = pickle.loads((run_dir / "Soft Core Mapping.ir_graph.pickle").read_bytes())

    class P:
        config = {
            **dp,
            **pc_res,
            "simulation_steps": pc["simulation_steps"],
            "device": device,
            "weight_bits": pc_res["weight_bits"],
            "input_shape": (1, 28, 28),
            "seed": 0,
            "firing_mode": dp["firing_mode"],
            "spike_generation_mode": dp["spike_generation_mode"],
            "thresholding_mode": dp["thresholding_mode"],
            "spiking_mode": dp["spiking_mode"],
            "max_simulation_samples": 200,
        }
        data_provider_factory = pf

    print("=== HCM (cached IR) ===")
    for sched in (True, False):
        pc_use = {**pc_res, "allow_scheduling": sched}
        P().config["allow_scheduling"] = sched
        hm = build_hybrid_mapping_for_pipeline(
            ir_graph, pc_use, pipeline_config=P().config
        )
        flow = build_spiking_hybrid_flow(P(), hm).to(device).eval()
        print(
            f"allow_scheduling={sched} stages={len(hm.stages)} "
            f"acc={accuracy(flow, device, pf):.4f}"
        )


if __name__ == "__main__":
    main()
