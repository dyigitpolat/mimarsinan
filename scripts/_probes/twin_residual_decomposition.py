"""PR17/17b: decompose the trajectory-dependent twin residual (calculus §13.4).

Per model (baseline vs anchored AQ cache), on identical batches:
  (1) per-hop twin-delta ledger: genuine decoded output vs the value-twin
      output per perceptron (systematic sign => convention Type-B);
  (2) per-hop tie-mass: distance of T*z/theta to the nearest staircase
      boundary (STE-parking => boundary-adjacent concentration);
  (3) identical grid-noise injection on the VALUE forward (+-theta/2T at every
      activation): margin sensitivity under the same eta.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import copy
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--batches", type=int, default=4)
args = parser.parse_args()

with open(f"{args.run}/_RUN_CONFIG/config.json") as f:
    cfg = json.load(f)
cfg["_working_directory"] = args.run

from mimarsinan.pipelining.session import PipelineSession

session = PipelineSession.from_config(cfg)
pipe = session.pipeline
pipe.load_cache()
pipe.set_up_requirements()
model = pipe.cache.get("Activation Quantization.model").to(pipe.config["device"])
manager = pipe.cache.get("Activation Quantization.adaptation_manager")
device = pipe.config["device"]
T = int(pipe.config["simulation_steps"])

from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

tuner = LIFAdaptationTuner(
    pipe, model=model, target_accuracy=pipe.get_target_metric(),
    lr=pipe.config["lr"], adaptation_manager=manager,
)
batches = [
    (x.to(device), y.to(device))
    for x, y in tuner.trainer.iter_validation_batches(args.batches)
]
print(f"[TRD] n={sum(y.numel() for _, y in batches)}", flush=True)

with tuner._finalize_flags_guard():
    clone = copy.deepcopy(tuner.model)
    tuner._force_full_transform_on_clone(clone)
perceptrons = list(clone.get_perceptrons())
thetas = [
    max(float(torch.as_tensor(p.activation_scale).detach().float().mean()), 1e-12)
    for p in perceptrons
]

# --- (2) tie-mass + (1) value-side per-hop outputs via hooks on one pass ---
def _sub(t: torch.Tensor, cap: int = 200_000) -> torch.Tensor:
    flat = t.detach().float().reshape(-1)
    return flat[:: max(1, flat.numel() // cap)].cpu()


pre, val_out = {}, {}
handles = []
for k, p in enumerate(perceptrons):
    def _hook(_m, inp, out, _k=k):
        pre.setdefault(_k, []).append(_sub(inp[0]))
        val_out.setdefault(_k, []).append(_sub(out))
    handles.append(p.activation.register_forward_hook(_hook))

correct_v = correct_g = correct_n = total = 0
with torch.no_grad():
    for x, y in batches:
        logits_v = clone(x)
        correct_v += int((logits_v.argmax(1) == y).sum())
        total += int(y.numel())
for h in handles:
    h.remove()

# --- (1) genuine per-hop decoded values + genuine accuracy, same batches ---
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver

driver = SegmentForwardDriver(clone.get_mapper_repr(), T, LifSegmentPolicy(retime=True))
gen_out = {}
with torch.no_grad():
    for x, y in batches:
        rec = {}
        logits_g = driver(x, node_value_recorder=rec)
        correct_g += int((logits_g.argmax(1) == y).sum())
        for k, p in enumerate(perceptrons):
            v = rec.get(id(p))
            if v is not None:
                gen_out.setdefault(k, []).append(_sub(v))

# --- (3) identical grid-noise injection on the value forward ---
noise_handles = []
for k, p in enumerate(perceptrons):
    def _noise(_m, _inp, out, _k=k):
        a = thetas[_k] / (2 * T)
        return out + torch.empty_like(out).uniform_(-a, a)
    noise_handles.append(p.activation.register_forward_hook(_noise))
torch.manual_seed(1)
with torch.no_grad():
    for x, y in batches:
        logits_n = clone(x)
        correct_n += int((logits_n.argmax(1) == y).sum())
for h in noise_handles:
    h.remove()

print(f"[TRD] acc value={correct_v/total:.4f} genuine={correct_g/total:.4f} "
      f"value+gridnoise={correct_n/total:.4f}", flush=True)
print(f"{'hop':>4} {'theta':>7} {'tie05':>6} {'tie01':>6} {'park+':>6} "
      f"{'d_mean/th':>9} {'d_abs/th':>8}")
for k in range(len(perceptrons)):
    z = torch.cat(pre[k])
    frac = (z * T / thetas[k]).frac()
    d = torch.minimum(frac, 1 - frac)
    tie05 = float((d < 0.05).float().mean())
    tie01 = float((d < 0.01).float().mean())
    park_above = float(((frac > 0.0) & (frac < 0.05)).float().mean())
    if k in gen_out and k in val_out:
        g = torch.cat(gen_out[k])
        v = torch.cat(val_out[k])
        n = min(g.numel(), v.numel())
        delta = (g[:n] - v[:n]) / thetas[k]
        dm, da = float(delta.mean()), float(delta.abs().mean())
    else:
        dm = da = float("nan")
    print(f"{k:>4} {thetas[k]:>7.3f} {tie05:>6.3f} {tie01:>6.3f} "
          f"{park_above:>6.3f} {dm:>9.4f} {da:>8.4f}", flush=True)
