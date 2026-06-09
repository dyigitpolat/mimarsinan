#!/usr/bin/env python3
"""Compare logits: native model vs SpikingHybridCoreFlow."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mimarsinan.data_handling.data_providers.mnist_data_provider  # noqa: F401
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import _ChipAlignedNFForward


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

    loader = DataLoaderFactory(pf).create_test_loader(8, pf.create())
    loader.num_workers = 0
    x, y = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        if isinstance(model.forward, _ChipAlignedNFForward):
            out_nf = model(x)
            native_fwd = model.forward._unpatched_forward
            out_native = native_fwd(x)
        else:
            out_nf = model(x)
            out_native = out_nf

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
        }

    hm = build_hybrid_mapping_for_pipeline(ir_graph, pc_res, pipeline_config=P.config)
    flow = build_spiking_hybrid_flow(P(), hm).to(device).eval()
    with torch.no_grad():
        out_hcm = flow(x)

    print("NF logits sample:", out_nf[0, :5].cpu(), "pred", out_nf[0].argmax().item(), "label", y[0].item())
    print("native logits sample:", out_native[0, :5].cpu(), "pred", out_native[0].argmax().item())
    print("HCM logits sample:", out_hcm[0, :5].cpu(), "pred", out_hcm[0].argmax().item())
    print("NF vs HCM max abs diff:", (out_nf - out_hcm).abs().max().item())
    print("native vs HCM max abs diff:", (out_native - out_hcm).abs().max().item())
    batch_acc_nf = (out_nf.argmax(1) == y.to(device)).float().mean().item()
    batch_acc_hcm = (out_hcm.argmax(1) == y.to(device)).float().mean().item()
    print(f"batch acc NF={batch_acc_nf:.2f} HCM={batch_acc_hcm:.2f}")


if __name__ == "__main__":
    main()
