#!/usr/bin/env python3
"""Compare Unified vs Hybrid TTFS logits on one batch."""
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
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.pipelining.simulation_factory import build_spiking_hybrid_flow


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
    hm = pickle.loads((run_dir / "hybrid_mapping.pickle").read_bytes())

    loader = DataLoaderFactory(pf).create_test_loader(16, pf.create())
    loader.num_workers = 0
    x, y = next(iter(loader))
    x = x.to(device)

    unified = SpikingUnifiedCoreFlow(
        (1, 28, 28),
        ir_graph,
        int(pc["simulation_steps"]),
        nn.Identity(),
        dp["firing_mode"],
        dp["spike_generation_mode"],
        dp["thresholding_mode"],
        spiking_mode="ttfs",
    ).to(device).eval()

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
            "spiking_mode": "ttfs",
        }

    hybrid = build_spiking_hybrid_flow(P(), hm).to(device).eval()

    with torch.no_grad():
        u = unified(x)
        h = hybrid(x)
        h_no_t = h / float(pc["simulation_steps"])

    print("unified pred", u[:4].argmax(1).tolist(), "labels", y[:4].tolist())
    print("hybrid pred", h[:4].argmax(1).tolist())
    print("hybrid/T pred", h_no_t[:4].argmax(1).tolist())
    print("max |u - h|", (u - h).abs().max().item())
    print("max |u - h/T|", (u - h_no_t).abs().max().item())
    acc_u = (u.argmax(1) == y.to(device)).float().mean().item()
    acc_h = (h.argmax(1) == y.to(device)).float().mean().item()
    acc_ht = (h_no_t.argmax(1) == y.to(device)).float().mean().item()
    print(f"batch acc unified={acc_u:.2f} hybrid={acc_h:.2f} hybrid/T={acc_ht:.2f}")


if __name__ == "__main__":
    main()
