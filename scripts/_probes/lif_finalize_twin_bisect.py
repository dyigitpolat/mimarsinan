"""F1 / PR1-PR2 (spiking_deployment_calculus.md §8): bisect the LIF finalize-twin
0.0115-vs-0.60 anomaly on the cached sigma-armed AQ run, through the REAL
pipeline + tuner machinery (probe == deploy by construction).

Arms (pipeline-faithful = retime=T, cycle-trains=T, manager-rebuild=T):
  R0  blended read at entry (no transform)      expect ~0.774 (analytic anchor)
  R1  faithful full-transform repro             expect ~0.011 +- 0.02 @ n=512
  R2  retime OFF, else faithful                 isolates the retime transcode
  R3  cycle-trains OFF, else faithful           isolates train propagation
  R4  manager-rebuild OFF (fresh LIF), retime T isolates update_activation
  R5  probe-replica (fresh LIF, retime F)       expect ~0.30 (ties to §10j)

Usage: python scripts/_probes/lif_finalize_twin_bisect.py [--arms R0,R1] [--eval-batches 8]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = "generated/t2_04_lif_vit_wq_s32_sched_offload_pruned_phased_deployment_run"

parser = argparse.ArgumentParser()
parser.add_argument("--arms", default="R0,R1")
parser.add_argument("--eval-batches", type=int, default=8)
parser.add_argument("--run", default=RUN)
args = parser.parse_args()
arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

with open(f"{args.run}/_RUN_CONFIG/config.json") as f:
    cfg = json.load(f)
# The persisted config strips underscore keys; pin the session to --run
# (otherwise the name-derived default silently points at the baseline dir).
cfg["_working_directory"] = args.run

from mimarsinan.pipelining.session import PipelineSession

t0 = time.time()
session = PipelineSession.from_config(cfg)
pipe = session.pipeline
pipe.load_cache()
pipe.set_up_requirements()
print(f"[BISECT] session+cache ready ({time.time()-t0:.0f}s)", flush=True)

model = pipe.cache.get("Activation Quantization.model")
manager = pipe.cache.get("Activation Quantization.adaptation_manager")
assert model is not None and manager is not None, "AQ cache entries missing"
device = pipe.config["device"]
model = model.to(device)

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.tuning.orchestration.mbh_ledger import (
    _measurement_guard,
    full_transform_measurement,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

t0 = time.time()
tuner = LIFAdaptationTuner(
    pipe,
    model=model,
    target_accuracy=pipe.get_target_metric(),
    lr=pipe.config["lr"],
    adaptation_manager=manager,
)
tuner._budget.eval_n_batches = int(args.eval_batches)
tuner._budget.progress_eval_batches = int(args.eval_batches)
print(
    f"[BISECT] tuner constructed ({time.time()-t0:.0f}s); "
    f"eval_n_batches={args.eval_batches} retiming={tuner._per_hop_retiming} "
    f"cycle_accurate={tuner._cycle_accurate} exact_qat={tuner._adaptation_plan.exact_qat}",
    flush=True,
)

T = int(pipe.config["simulation_steps"])
results = {}


def read_full(label):
    t = time.time()
    value = float(full_transform_measurement(tuner))
    print(f"[BISECT] {label}: {value:.4f}  ({time.time()-t:.0f}s)", flush=True)
    return value


def fresh_lif_force(clone):
    """R4/R5: the §10j probe's install — fresh LIF at the perceptron's own
    scale, decorators dropped, NO manager rebuild, NO blend-rate change."""
    for p in clone.get_perceptrons():
        scale = float(torch.as_tensor(p.activation_scale).detach().float().mean())
        p.activation = LIFActivation(
            T=T,
            activation_scale=torch.tensor(max(scale, 1e-4)),
            thresholding_mode="<",
            firing_mode="Default",
            bias_mode="on_chip",
        ).to(device)


def no_cycle_trains_after(model=None):
    from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

    apply_cycle_accurate_trains_to_model(
        tuner.model if model is None else model, False,
    )


for arm in arms:
    if arm == "R0":
        with _measurement_guard(tuner.trainer):
            t = time.time()
            value = float(tuner.trainer.validate_n_batches(int(args.eval_batches)))
        print(f"[BISECT] R0 blended entry (no transform): {value:.4f}  ({time.time()-t:.0f}s)", flush=True)
        results[arm] = value
    elif arm == "R1":
        results[arm] = read_full("R1 faithful (retime=T trains=T rebuild=T)")
    elif arm == "R2":
        tuner._per_hop_retiming = False
        results[arm] = read_full("R2 retime OFF (trains=T rebuild=T)")
        tuner._per_hop_retiming = True
    elif arm == "R3":
        tuner._after_finalize_rebuild = no_cycle_trains_after
        results[arm] = read_full("R3 cycle-trains OFF (retime=T rebuild=T)")
        tuner.__dict__.pop("_after_finalize_rebuild")
    elif arm == "R4":
        tuner._force_full_transform_on_clone = fresh_lif_force
        results[arm] = read_full("R4 rebuild OFF / fresh LIF (retime=T)")
        tuner.__dict__.pop("_force_full_transform_on_clone")
    elif arm == "R5":
        tuner._force_full_transform_on_clone = fresh_lif_force
        tuner._per_hop_retiming = False
        results[arm] = read_full("R5 probe replica (fresh LIF, retime=F)")
        tuner._per_hop_retiming = True
        tuner.__dict__.pop("_force_full_transform_on_clone")
    else:
        print(f"[BISECT] unknown arm {arm!r}", flush=True)

print(f"[BISECT] RESULTS {json.dumps(results)}", flush=True)
