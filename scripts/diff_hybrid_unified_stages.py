#!/usr/bin/env python3
"""Find first hybrid stage diverging from unified TTFS on a saved run."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mimarsinan.data_handling.data_providers.mnist_data_provider  # noqa: F401
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_torch, resolve_stage_compute_scales
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow, _COMPUTE_DTYPE
from mimarsinan.models.spiking.unified.flow import SpikingUnifiedCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow


def main() -> None:
    run_dir = Path(sys.argv[1])
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

    loader = DataLoaderFactory(pf).create_test_loader(4, pf.create())
    loader.num_workers = 0
    x, _ = next(iter(loader))
    x = x.to(device).view(x.shape[0], -1)

    unified = SpikingUnifiedCoreFlow(
        (1, 28, 28), ir_graph, int(pc["simulation_steps"]), nn.Identity(),
        dp["firing_mode"], dp["spike_generation_mode"], dp["thresholding_mode"],
        spiking_mode="ttfs",
    ).to(device).eval()

    u_cache: dict[int, torch.Tensor] = {}
    x_c = x.to(unified._compute_dtype)
    batch = x.shape[0]

    for node in unified.nodes:
        if isinstance(node, NeuralCore):
            w = unified._get_weight(node).to(unified._compute_dtype)
            th = unified._get_threshold(node).to(unified._compute_dtype)
            hb = unified._get_hw_bias(node)
            spans = unified._input_spans[int(node.id)]
            in_dim = len(node.input_sources.flatten())
            inp = torch.zeros(batch, in_dim, device=device, dtype=unified._compute_dtype)
            unified._fill_activation_from_ir_spans(
                inp, x=x_c, activation_cache=u_cache, spans=spans,
            )
            out = torch.matmul(w, inp.T).T
            if hb is not None:
                out = out + hb.to(unified._compute_dtype)
            out = torch.relu(out) / th.clamp(min=1e-12)
            u_cache[node.id] = out.clamp(0, 1)
        elif isinstance(node, ComputeOp):
            u_cache[node.id] = unified._execute_compute_op_ttfs(
                node, x, batch, device, u_cache,
            )

    class P:
        config = {**dp, **pc_res, "simulation_steps": pc["simulation_steps"], "device": device,
                  "weight_bits": pc_res["weight_bits"], "input_shape": (1, 28, 28),
                  "firing_mode": dp["firing_mode"], "spike_generation_mode": dp["spike_generation_mode"],
                  "thresholding_mode": dp["thresholding_mode"], "spiking_mode": "ttfs",
                  }

    hcm = build_spiking_hybrid_flow(P(), hm).to(device).eval()
    h_state: dict[int, torch.Tensor] = {-2: x.to(_COMPUTE_DTYPE)}

    for si, stage in enumerate(hm.stages):
        if stage.kind == "neural":
            seg_in = hcm._assemble_segment_input(stage.input_map, h_state, batch, device)
            seg_out = hcm._run_neural_segment_ttfs(stage, input_activations=seg_in, quantized=False)
            hcm._store_segment_output(stage.output_map, h_state, seg_out)
            # Compare each output slice to unified node
            for s in stage.output_map:
                u_act = u_cache.get(s.node_id)
                h_act = h_state[s.node_id][:, s.offset : s.offset + s.size]
                if u_act is None:
                    print(f"stage {si} neural {stage.name}: node {s.node_id} missing in unified")
                    continue
                u_slice = u_act[:, s.offset : s.offset + s.size]
                if u_slice.numel() == 0 or h_act.numel() == 0:
                    print(
                        f"SIZE stage {si} node {s.node_id} "
                        f"u={tuple(u_slice.shape)} h={tuple(h_act.shape)}"
                    )
                    continue
                diff = (u_slice.double() - h_act.double()).abs().max().item()
                if diff > 1e-4:
                    print(
                        f"DIVERGE stage {si} neural {stage.name!r} "
                        f"node {s.node_id} offset={s.offset} max_diff={diff:.6f}"
                    )
        else:
            op = stage.compute_op
            assert op is not None
            in_s, out_s = resolve_stage_compute_scales(hm, op.id, apply_ttfs=True)
            h_state[op.id] = execute_compute_op_torch(
                op, x, h_state, in_scale=in_s, out_scale=out_s, output_dtype=_COMPUTE_DTYPE,
            )
            u_act = u_cache.get(op.id)
            if u_act is not None:
                diff = (u_act.double() - h_state[op.id].double()).abs().max().item()
                if diff > 1e-4:
                    print(
                        f"DIVERGE stage {si} compute {op.name!r} id={op.id} "
                        f"max_diff={diff:.6f} in_scale={in_s} out_scale={out_s}"
                    )

    print("done")


if __name__ == "__main__":
    main()
