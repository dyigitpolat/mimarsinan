"""PR19': cancel the same-sign per-hop genuine bias on the anchored model with
the EXISTING first-moment machinery (lif_distribution_matching), no training.

Prediction (calculus §14): DFQ mean-matching to the ORIGIN teacher lifts the
anchored model's genuine read from ~0.715 toward the analytic band.
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

from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.lif_distribution_matching import (
    match_lif_activation_distributions,
)
from mimarsinan.tuning.teacher import find_reference_teacher, freeze_module
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

tuner = LIFAdaptationTuner(
    pipe, model=model, target_accuracy=pipe.get_target_metric(),
    lr=pipe.config["lr"], adaptation_manager=manager,
)
batches = [
    (x.to(device), y.to(device))
    for x, y in tuner.trainer.iter_validation_batches(8)
]
with tuner._finalize_flags_guard():
    clone = copy.deepcopy(tuner.model)
    tuner._force_full_transform_on_clone(clone)

teacher = find_reference_teacher(pipe)
assert teacher is not None, "origin teacher must be cached"
teacher = freeze_module(teacher.to(device))


def genuine_acc(n_batches: int = 4) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in batches[:n_batches]:
            logits = chip_aligned_segment_forward(clone, x, T, retime=True)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


entry = genuine_acc(8)
print(f"[PR19] genuine BEFORE distmatch (n=512): {entry:.4f}", flush=True)

cal_x = torch.cat([x for x, _ in batches[:8]])
stats = match_lif_activation_distributions(
    clone, teacher, cal_x, T,
    bias_iters=10, eta=0.5, probe=lambda: genuine_acc(4), probe_patience=3,
)
print(f"[PR19] distmatch stats: { {k: round(float(v), 4) for k, v in stats.items() if isinstance(v, (int, float))} }", flush=True)
final = genuine_acc(8)
print(f"[PR19] genuine AFTER distmatch (n=512): {final:.4f}  (delta {final-entry:+.4f})", flush=True)
