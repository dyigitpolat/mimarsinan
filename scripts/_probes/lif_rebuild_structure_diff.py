"""F1 follow-up: WHAT does the lif_active finalize rebuild install, vs a fresh
LIFActivation at the same theta? Structural module tree + per-layer behavior
diff on a synthetic sweep — localizes the 0.29->0.0156 rebuild defect.
"""
import copy
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = "generated/t2_04_lif_vit_wq_s32_sched_offload_pruned_phased_deployment_run"
with open(f"{RUN}/_RUN_CONFIG/config.json") as f:
    cfg = json.load(f)

from mimarsinan.pipelining.session import PipelineSession

session = PipelineSession.from_config(cfg)
pipe = session.pipeline
pipe.load_cache()
pipe.set_up_requirements()
model = pipe.cache.get("Activation Quantization.model").to(pipe.config["device"])
manager = pipe.cache.get("Activation Quantization.adaptation_manager")

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

tuner = LIFAdaptationTuner(
    pipe, model=model, target_accuracy=pipe.get_target_metric(),
    lr=pipe.config["lr"], adaptation_manager=manager,
)
device = pipe.config["device"]
T = int(pipe.config["simulation_steps"])

with tuner._finalize_flags_guard():
    clone = copy.deepcopy(tuner.model)
    tuner._force_full_transform_on_clone(clone)

perceptrons = list(clone.get_perceptrons())
print(f"\n=== rebuilt activation structure ({len(perceptrons)} perceptrons) ===")
for k in (0, 1, 5, 11):
    p = perceptrons[k]
    print(f"\n--- perceptron[{k}] activation tree ---")
    print(repr(p.activation)[:1200])
    print(f"    input_activation: {repr(p.input_activation)[:300]}")

print("\n=== behavior diff on synthetic sweep (rebuilt vs fresh LIF, same theta) ===")
torch.manual_seed(0)
for k in range(len(perceptrons)):
    p = perceptrons[k]
    theta = float(torch.as_tensor(p.activation_scale).detach().float().mean())
    theta = max(theta, 1e-6)
    z = torch.linspace(-theta, 2 * theta, 4096, device=device).reshape(64, 64)
    fresh = LIFActivation(
        T=T, activation_scale=torch.tensor(theta), thresholding_mode="<",
        firing_mode="Default", bias_mode="on_chip",
    ).to(device)
    with torch.no_grad():
        y_rebuilt = p.activation(z)
        y_fresh = fresh(z)
    d = (y_rebuilt - y_fresh).abs()
    print(
        f"  p[{k:02d}] theta={theta:8.4f} mean|d|={float(d.mean()):.5f} "
        f"max|d|={float(d.max()):.5f} grid={theta/T:.5f} "
        f"rebuilt_out_range=[{float(y_rebuilt.min()):.3f},{float(y_rebuilt.max()):.3f}] "
        f"fresh_out_range=[{float(y_fresh.min()):.3f},{float(y_fresh.max()):.3f}]"
    )
