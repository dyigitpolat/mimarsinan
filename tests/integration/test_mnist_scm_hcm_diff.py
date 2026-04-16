"""Diff SCM vs HCM on the actual MNIST IR graph from the latest pipeline run.

Loads ``generated/mnist_hard_all_scm_profile/Soft Core Mapping.ir_graph.pickle``
(produced by the real pipeline) and runs both ``SpikingUnifiedCoreFlow`` (SCM)
and ``SpikingHybridCoreFlow`` (HCM) on identical input.  Reports the
per-NeuralCore-output divergence in graph-execution order so we can pinpoint
the first node where they disagree.

This is a diagnostic, not a regression gate — it only runs when the pickle
exists (i.e. you've recently completed at least Soft Core Mapping on MNIST).
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


WORK_DIR = Path("generated/mnist_hard_all_scm_profile")
IR_PICKLE = WORK_DIR / "Soft Core Mapping.ir_graph.pickle"
PLATFORM_CFG = WORK_DIR / "Model Configuration.platform_constraints_resolved.json"


def _load_inputs(n_samples: int = 8) -> torch.Tensor:
    from mimarsinan.data_handling.data_providers.mnist_data_provider import (
        MNIST_DataProvider,
    )
    provider = MNIST_DataProvider("./datasets")
    test_ds = provider._get_test_dataset()
    xs = []
    for i in range(n_samples):
        x, _ = test_ds[i]
        xs.append(x.unsqueeze(0))
    return torch.cat(xs, dim=0)


@pytest.mark.skipif(not IR_PICKLE.exists(), reason=f"missing {IR_PICKLE}")
def test_mnist_scm_hcm_first_divergence():
    with open(IR_PICKLE, "rb") as f:
        ir_graph = pickle.load(f)

    # Inspect any encoding-ComputeOp module's decorator chain — a non-zero
    # NoisyDropout rate would inject batch-stochastic noise into both flows
    # and explain run-to-run accuracy drift.
    print("\n=== Encoding ComputeOp decorator inspection ===")
    for node in ir_graph.nodes:
        if isinstance(node, ComputeOp) and (node.params or {}).get("module") is not None:
            module = node.params["module"]
            # Conv2DPerceptronMapper wraps a Perceptron; PerceptronMapper *is* one.
            inner = getattr(module, "perceptron", None) or module
            act = getattr(inner, "activation", None)
            decorators = getattr(act, "decorators", None) if act is not None else None
            print(f"node {node.id} {node.name}: outer={type(module).__name__} "
                  f"inner={type(inner).__name__} "
                  f"activation={type(act).__name__ if act else None} "
                  f"n_decorators={len(decorators) if decorators else 0}")
            if decorators:
                for d in decorators:
                    rate = getattr(d, "rate", None)
                    inner = getattr(d, "decorator", None)
                    print(f"    {type(d).__name__} rate={rate} inner={type(inner).__name__ if inner else None}")
                    if inner is not None:
                        for attr in ("clamp_min", "clamp_max", "shift", "scale", "levels_before_c", "c"):
                            v = getattr(inner, attr, None)
                            if v is not None:
                                print(f"       inner.{attr}={v}")

    with open(PLATFORM_CFG) as f:
        platform = json.load(f)
    cores_config = platform["cores"]

    sim_length = 32
    input_shape = (1, 28, 28)
    spiking_mode = "ttfs_quantized"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = _load_inputs(256).to(device)

    # --- Build SCM and capture per-node activations ---
    scm = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, sim_length, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode=spiking_mode,
    ).eval().to(device)

    scm_cache: dict[int, torch.Tensor] = {}
    orig_compute = scm._execute_compute_op_ttfs
    orig_get_w = scm._get_weight
    orig_get_th = scm._get_threshold
    orig_get_hb = scm._get_hw_bias

    # Re-implement _forward_ttfs_quantized inline so we can snapshot every
    # NeuralCore's activation at the moment it's written into the cache.
    def scm_forward_capture(x):
        x = scm.preprocessor(x)
        x = x.view(x.shape[0], -1)
        S = scm.simulation_length
        ctype = scm._compute_dtype
        x_compute = x.to(ctype)
        cache: dict[int, torch.Tensor] = {}
        batch_size = x.shape[0]
        device = x.device
        for node_idx, node in enumerate(scm.nodes):
            if isinstance(node, NeuralCore):
                w = orig_get_w(node).to(ctype)
                th = orig_get_th(node).to(ctype)
                hb = orig_get_hb(node)
                spans = scm._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device, dtype=ctype)
                scm._fill_activation_from_ir_spans(inp, x=x_compute, activation_cache=cache, spans=spans)
                V = torch.matmul(w, inp.T).T
                if hb is not None:
                    V = V + hb.to(ctype)
                safe = th.clamp(min=1e-12)
                k_raw = torch.ceil(S * (1.0 - V / safe))
                fires = k_raw < S
                k = k_raw.clamp(0, S - 1)
                act = torch.where(fires, (S - k) / S, torch.zeros_like(k))
                cache[node.id] = act
                scm_cache[node.id] = act.detach().to(torch.float64).cpu().clone()
            elif isinstance(node, ComputeOp):
                cache[node.id] = orig_compute(node, x, batch_size, device, cache)
                scm_cache[node.id] = cache[node.id].detach().to(torch.float64).cpu().clone()
        return cache

    with torch.no_grad():
        scm_forward_capture(x)

    # --- Build HCM and capture per-segment outputs ---
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=cores_config,
        allow_neuron_splitting=bool(platform.get("allow_neuron_splitting", False)),
        allow_scheduling=bool(platform.get("allow_scheduling", False)),
        allow_coalescing=bool(platform.get("allow_coalescing", False)),
    )
    hcm = SpikingHybridCoreFlow(
        input_shape, hybrid, sim_length, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode=spiking_mode,
    ).eval().to(device)

    # Patch _store_segment_output to snapshot per-node outputs to dict.
    hcm_cache: dict[int, torch.Tensor] = {}
    orig_store = hcm._store_segment_output

    def store_capture(output_map, state_buffer, output_tensor):
        orig_store(output_map, state_buffer, output_tensor)
        for s in output_map:
            v = state_buffer[s.node_id]
            hcm_cache[s.node_id] = v.detach().to(torch.float64).cpu().clone()

    hcm._store_segment_output = store_capture

    # Also snapshot ComputeOp outputs.
    from mimarsinan.models.hybrid_core_flow import _COMPUTE_DTYPE
    orig_forward = hcm._forward_ttfs

    def hcm_forward_capture(x_flat):
        out = orig_forward(x_flat)
        # state_buffer is local to _forward_ttfs; reconstruct by re-running.
        return out

    with torch.no_grad():
        x_flat = x.view(x.shape[0], -1)
        # Manually walk the segments to capture compute op outputs too.
        hcm_state: dict[int, torch.Tensor] = {-2: x_flat.to(_COMPUTE_DTYPE)}
        out_scales = getattr(hybrid, "node_activation_scales", {})
        in_scales = getattr(hybrid, "node_input_activation_scales", out_scales)
        T = hcm.simulation_length
        bsz = x_flat.shape[0]
        device = x_flat.device
        for stage in hybrid.stages:
            if stage.kind == "neural":
                seg_input = hcm._assemble_segment_input(stage.input_map, hcm_state, bsz, device)
                seg_output = hcm._run_neural_segment_ttfs(
                    stage, input_activations=seg_input, quantized=True,
                )
                hcm._store_segment_output(stage.output_map, hcm_state, seg_output)
            else:
                op = stage.compute_op
                in_s = in_scales.get(op.id, 1.0)
                out_s = out_scales.get(op.id, 1.0)
                xf = x_flat.to(torch.float32)
                sbf = {k: v.to(torch.float32) for k, v in hcm_state.items()}
                gathered = op.gather_inputs(xf, sbf)
                if abs(in_s - 1.0) > 1e-9:
                    gathered = gathered * in_s
                result = op.execute_on_gathered(gathered)
                if abs(out_s - 1.0) > 1e-9:
                    result = result / out_s
                hcm_state[op.id] = result.to(_COMPUTE_DTYPE)
                hcm_cache[op.id] = result.detach().to(torch.float64).cpu().clone()

    # --- Compare per-node ---
    common = sorted(set(scm_cache.keys()) & set(hcm_cache.keys()),
                    key=lambda nid: next(i for i, n in enumerate(ir_graph.nodes)
                                         if n.id == nid))
    print(f"\n=== Per-node SCM vs HCM diff (common nodes: {len(common)}) ===")
    first_div = None
    for nid in common:
        s = scm_cache[nid]
        h = hcm_cache[nid]
        if s.shape != h.shape:
            print(f"  node {nid}: shape mismatch SCM {s.shape} vs HCM {h.shape}")
            continue
        diff = (s - h).abs().max().item()
        node_name = next(n.name for n in ir_graph.nodes if n.id == nid)
        node_kind = type(next(n for n in ir_graph.nodes if n.id == nid)).__name__
        marker = "  ✱" if diff > 1e-6 else "   "
        print(f"{marker}node {nid:4d} [{node_kind:11s}] {node_name[:40]:40s} "
              f"max_diff={diff:.3e} shape={tuple(s.shape)}")
        if diff > 1e-6 and first_div is None:
            first_div = (nid, node_name, node_kind, diff)

    if first_div:
        print(f"\nFirst divergence: node id={first_div[0]} name={first_div[1]} "
              f"kind={first_div[2]} max_diff={first_div[3]:.3e}")
    else:
        print("\nAll common nodes match within 1e-6.")

    # --- End-to-end accuracy check via BasicTrainer (mimics pipeline) ---
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
    from mimarsinan.model_training.basic_trainer import BasicTrainer

    # Force registry population.
    import mimarsinan.data_handling.data_providers  # noqa: F401
    factory = BasicDataProviderFactory("MNIST_DataProvider", "./datasets")

    scm_trainer = BasicTrainer(scm, device, DataLoaderFactory(factory), None)
    # Mimic SCM step: cap test_batch_size at 8.
    scm_trainer.set_test_batch_size(min(scm_trainer.test_batch_size, 8))
    scm_acc = scm_trainer.test(max_batches=None)
    print(f"\nSCM end-to-end accuracy (batch=8): {scm_acc}")

    hcm_trainer = BasicTrainer(hcm, device, DataLoaderFactory(factory), None)
    hcm_acc = hcm_trainer.test(max_batches=None)
    print(f"HCM end-to-end accuracy (default batch={hcm_trainer.test_batch_size}): {hcm_acc}")

    hcm_trainer_b8 = BasicTrainer(hcm, device, DataLoaderFactory(factory), None)
    hcm_trainer_b8.set_test_batch_size(8)
    hcm_acc_b8 = hcm_trainer_b8.test(max_batches=None)
    print(f"HCM end-to-end accuracy (batch=8): {hcm_acc_b8}")

    # Also try SCM at default (HCM-equivalent) batch size.
    scm_trainer2 = BasicTrainer(scm, device, DataLoaderFactory(factory), None)
    scm_acc_big = scm_trainer2.test(max_batches=None)
    print(f"SCM end-to-end accuracy (default batch={scm_trainer2.test_batch_size}): {scm_acc_big}")

    # Self-consistency: run SCM twice on the same input.
    print("\n=== SCM self-consistency: same input, two forwards ===")
    test_ds = factory.create()._get_test_dataset()
    samples_for_self = torch.cat([test_ds[i][0].unsqueeze(0) for i in range(8)], dim=0).to(device)
    with torch.no_grad():
        a1 = scm(samples_for_self).to(torch.float64)
        a2 = scm(samples_for_self).to(torch.float64)
    print(f"  forward1 vs forward2 max_diff: {(a1-a2).abs().max().item():.3e}")

    # Compare argmax disagreement between batch=8 and batch=256 over a wider window.
    print("\n=== Argmax disagreement count: batch=8 vs batch=256 over 256 samples ===")
    test_ds = factory.create()._get_test_dataset()
    samples = torch.cat([test_ds[i][0].unsqueeze(0) for i in range(256)], dim=0).to(device)

    with torch.no_grad():
        out_big = scm(samples).to(torch.float64)
        per_sample_outs = []
        for start in range(0, 256, 8):
            chunk = samples[start:start+8]
            per_sample_outs.append(scm(chunk).to(torch.float64))
        out_small = torch.cat(per_sample_outs, dim=0)

    argmax_big = out_big.argmax(dim=1)
    argmax_small = out_small.argmax(dim=1)
    disagreements = int((argmax_big != argmax_small).sum().item())
    print(f"argmax disagreements: {disagreements} / 256 = {disagreements/256:.1%}")
    if disagreements:
        idxs = (argmax_big != argmax_small).nonzero().flatten()[:5].tolist()
        for i in idxs:
            d_i = (out_big[i] - out_small[i]).abs().max().item()
            print(f"  sample {i}: argmax(big)={argmax_big[i].item()} argmax(small)={argmax_small[i].item()} max_diff={d_i:.3e}")
