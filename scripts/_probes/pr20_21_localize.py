"""G1 (calculus §14.4): localize the same-sign twin drift.

PR20 — token-resolved signed per-hop deltas (CLS row vs patch mean): the
classifier reads only CLS; pooled stats dilute the damage that matters.
PR21/V1 — insert the seam κ_T-grid round into the VALUE twin at every entry:
V1 ≈ V0 => the drift lives in per-cycle cascade physics; V1 craters => the
seam round convention is the term.
"""
import copy
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    "generated/t2_04_originanchor_ab_phased_deployment_run"
)
with open(f"{RUN}/_RUN_CONFIG/config.json") as f:
    cfg = json.load(f)
cfg["_working_directory"] = RUN

from mimarsinan.pipelining.session import PipelineSession

session = PipelineSession.from_config(cfg)
pipe = session.pipeline
pipe.load_cache()
pipe.set_up_requirements()
model = pipe.cache.get("Activation Quantization.model").to(pipe.config["device"])
manager = pipe.cache.get("Activation Quantization.adaptation_manager")
device = pipe.config["device"]
T = int(pipe.config["simulation_steps"])

from mimarsinan.spiking.scale_aware_boundaries import (
    read_boundary_out_scales,
    stamped_input_boundary_scale,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

tuner = LIFAdaptationTuner(
    pipe, model=model, target_accuracy=pipe.get_target_metric(),
    lr=pipe.config["lr"], adaptation_manager=manager,
)
batches = [
    (x.to(device), y.to(device))
    for x, y in tuner.trainer.iter_validation_batches(4)
]
with tuner._finalize_flags_guard():
    clone = copy.deepcopy(tuner.model)
    tuner._force_full_transform_on_clone(clone)
perceptrons = list(clone.get_perceptrons())
thetas = [
    max(float(torch.as_tensor(p.activation_scale).detach().float().mean()), 1e-12)
    for p in perceptrons
]
repr_ = clone.get_mapper_repr()
table = read_boundary_out_scales(
    repr_, input_data_scale=stamped_input_boundary_scale(repr_),
)
entry_kappas = [
    max(float(torch.as_tensor(p.input_activation_scale).detach().float().mean()), 1e-12)
    for p in perceptrons
]

# --- V0 value forward + per-hop value outputs (token-shaped) ---
val_out = {}
handles = []
for k, p in enumerate(perceptrons):
    def _hook(_m, _inp, out, _k=k):
        val_out.setdefault(_k, []).append(out.detach().float().cpu())
    handles.append(p.activation.register_forward_hook(_hook))
correct_v0 = total = 0
with torch.no_grad():
    for x, y in batches:
        logits = clone(x)
        correct_v0 += int((logits.argmax(1) == y).sum())
        total += int(y.numel())
for h in handles:
    h.remove()

# --- V1: value forward + seam grid round at every entry ---
def _make_round(kappa):
    def _pre(_m, inp):
        x = inp[0]
        return (torch.round((x / kappa).clamp(0.0, 1.0) * T) / T * kappa,)
    return _pre

round_handles = [
    p.input_activation.register_forward_pre_hook(_make_round(entry_kappas[k]))
    for k, p in enumerate(perceptrons)
]
correct_v1 = 0
with torch.no_grad():
    for x, y in batches:
        logits = clone(x)
        correct_v1 += int((logits.argmax(1) == y).sum())
for h in round_handles:
    h.remove()

# --- genuine walk + per-hop decoded outputs (token-shaped) ---
driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(retime=True))
gen_out = {}
correct_g = 0
with torch.no_grad():
    for x, y in batches:
        rec = {}
        logits = driver(x, node_value_recorder=rec)
        correct_g += int((logits.argmax(1) == y).sum())
        for k, p in enumerate(perceptrons):
            v = rec.get(id(p))
            if v is not None:
                gen_out.setdefault(k, []).append(v.detach().float().cpu())

print(f"[G1] n={total}  V0(value)={correct_v0/total:.4f}  "
      f"V1(value+seam-round)={correct_v1/total:.4f}  genuine={correct_g/total:.4f}",
      flush=True)
print(f"{'hop':>4} {'theta':>7} {'cls_dmean/th':>12} {'patch_dmean/th':>14} "
      f"{'cls_dabs/th':>11} {'patch_dabs/th':>13}")
for k in range(len(perceptrons)):
    if k not in gen_out or k not in val_out:
        continue
    g = torch.cat(gen_out[k])   # (B, tokens, feat)
    v = torch.cat(val_out[k])
    if g.dim() != 3 or v.shape != g.shape:
        print(f"{k:>4} shape mismatch g={tuple(g.shape)} v={tuple(v.shape)}")
        continue
    d = (g - v) / thetas[k]
    cls_d, patch_d = d[:, 0, :], d[:, 1:, :]
    print(f"{k:>4} {thetas[k]:>7.3f} {float(cls_d.mean()):>12.4f} "
          f"{float(patch_d.mean()):>14.4f} {float(cls_d.abs().mean()):>11.4f} "
          f"{float(patch_d.abs().mean()):>13.4f}", flush=True)
