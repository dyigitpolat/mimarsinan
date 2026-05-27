#!/usr/bin/env python3
"""Compare Unified vs Hybrid accuracy by spiking_mode on a saved SCM run."""
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
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.unified.flow import SpikingUnifiedCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "generated/mnist_hard_all_lif_ca60_phased_deployment_run_20260522_121635"
    )
    cfg = json.loads((run_dir / "_RUN_CONFIG/config.json").read_text())
    pc_res = json.loads(
        (run_dir / "Model Configuration.platform_constraints_resolved.json").read_text()
    )
    pc = cfg["platform_constraints"]
    dp = cfg["deployment_parameters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf = BasicDataProviderFactory(cfg["data_provider_name"], "")

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
            "max_simulation_samples": 200,
        }
        data_provider_factory = pf

    def acc_flow(flow) -> float:
        t = BasicTrainer(flow, device, DataLoaderFactory(pf), None)
        t.train_loader.num_workers = 0
        t.test_loader.num_workers = 0
        return float(t.test_on_subsample(max_samples=200, seed=0))

    u = SpikingUnifiedCoreFlow(
        (1, 28, 28),
        ir_graph,
        int(pc["simulation_steps"]),
        nn.Identity(),
        dp["firing_mode"],
        dp["spike_generation_mode"],
        dp["thresholding_mode"],
        spiking_mode="ttfs",
    ).to(device).eval()
    print(f"Unified ttfs: {acc_flow(u):.4f}")

    hm = pickle.loads((run_dir / "hybrid_mapping.pickle").read_bytes())
    for mode in ("ttfs", "ttfs_quantized"):
        p = P()
        p.config["spiking_mode"] = mode
        flow = build_spiking_hybrid_flow(p, hm)
        print(f"Hybrid {mode}: {acc_flow(flow):.4f}")


if __name__ == "__main__":
    main()
