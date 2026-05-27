#!/usr/bin/env python3
"""Reproduce NF vs SCM metric on a saved run directory."""
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
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_spiking_flow_for_metric,
    build_spiking_hybrid_flow,
    run_hcm_spiking_test,
)


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "generated/mnist_hard_all_lif_phased_deployment_run_20260520_084932"
    )
    cfg_json = json.loads((run_dir / "_RUN_CONFIG/config.json").read_text())
    pc_res = json.loads(
        (run_dir / "Model Configuration.platform_constraints_resolved.json").read_text()
    )
    pc = cfg_json["platform_constraints"]
    dp = cfg_json["deployment_parameters"]

    class PipelineStub:
        config = {
            **dp,
            **pc_res,
            "simulation_steps": pc["simulation_steps"],
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "data_provider_name": cfg_json["data_provider_name"],
            "weight_bits": pc_res["weight_bits"],
            "input_shape": (1, 28, 28),
            "seed": int(cfg_json.get("seed", 0)),
            "firing_mode": dp["firing_mode"],
            "spike_generation_mode": dp["spike_generation_mode"],
            "thresholding_mode": dp["thresholding_mode"],
            "spiking_mode": dp["spiking_mode"],
            "max_simulation_samples": int(dp.get("max_simulation_samples", 500)),
            "simulation_batch_count": 1,
        }
        cache = type("C", (), {"add": lambda *a, **k: None})()
        data_provider_factory = BasicDataProviderFactory(
            cfg_json["data_provider_name"], ""
        )

    ir_graph = pickle.loads((run_dir / "Soft Core Mapping.ir_graph.pickle").read_bytes())

    p = PipelineStub()
    hm = build_hybrid_mapping_for_pipeline(
        ir_graph, pc_res, pipeline_config=p.config
    )
    flow = build_spiking_flow_for_metric(p, hm, ir_graph)
    acc = run_hcm_spiking_test(
        p, flow, device=p.config["device"], max_batch_cap=32
    )
    print(f"rebuild metric acc={acc:.4f} stages={len(hm.stages)}")

    obj = torch.load(
        run_dir / "Normalization Fusion.fused_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = obj[0] if isinstance(obj, tuple) else obj
    model.eval()
    pre = getattr(model, "preprocessor", None)
    print(f"model type={type(model).__name__} preprocessor={type(pre).__name__ if pre else None}")

    p = PipelineStub()
    hybrid = pickle.loads((run_dir / "hybrid_mapping.pickle").read_bytes())
    for label, pre_mod in [("no_preprocessor", None), ("with_preprocessor", pre)]:
        flow = build_spiking_hybrid_flow(p, hybrid, preprocessor=pre_mod)
        acc = run_hcm_spiking_test(p, flow, device=p.config["device"], max_batch_cap=32)
        print(f"HCM {label} acc={acc:.4f}")
    trainer = BasicTrainer(
        model, p.config["device"], DataLoaderFactory(p.data_provider_factory), None
    )
    trainer.train_loader.num_workers = 0
    trainer.test_loader.num_workers = 0
    nf = trainer.test_on_subsample(max_samples=500, seed=0)
    print(f"NF subsample acc={nf:.4f}")


if __name__ == "__main__":
    main()
