"""Summarize a deployment run: ANN target, per-step wall+metric, deployed gap, FAST budget.

Usage: python experiments/summarize.py <experiment_name>   (reads generated/<name>_phased_deployment_run)
       python experiments/summarize.py --dir <generated_run_dir>
"""
import json, os, sys

# Steps that constitute the FAST "tuning/transformation" budget (calibration +
# fine-tune + quantization + fusion); excludes one-time ANN Pretraining and the
# Torch Mapping import, and the deployment mapping/sim.
FAST_STEPS = {
    "Activation Analysis", "LIF Adaptation", "TTFS Cycle Fine-Tuning",
    "Activation Quantization", "Clamp Adaptation", "Weight Quantization",
    "Quantization Verification", "Normalization Fusion", "Pruning",
}
ANN_STEP = "Pretraining"
# Lossless gate = Soft Core Mapping (FULL test set via deployment_metric_full_eval).
# The nevresim "Simulation" step is subsampled (max_simulation_samples) -> noisy, NOT the gate.
DEPLOY_STEP = "Soft Core Mapping"
NEVRESIM_STEP = "Simulation"


def summarize(run_dir):
    sp = os.path.join(run_dir, "_GUI_STATE", "steps.json")
    if not os.path.exists(sp):
        print(f"no steps.json at {sp}"); return
    steps = json.load(open(sp))["steps"]
    ann = None; deployed = None; nf = None; nevresim = None
    fast_wall = 0.0; total_wall = 0.0
    print(f"{'step':30} {'wall(s)':>9} {'metric':>9}")
    for name, info in steps.items():
        wall = info.get("end_time", 0) - info.get("start_time", 0)
        m = info.get("target_metric")
        total_wall += wall
        if name in FAST_STEPS:
            fast_wall += wall
        if name == ANN_STEP and m:
            ann = m
        if name == "Normalization Fusion" and m:
            nf = m
        if name == DEPLOY_STEP and m:
            deployed = m
        if name == NEVRESIM_STEP and m:
            nevresim = m
        print(f"{name:30} {wall:9.1f} {('' if m is None else f'{m:.4f}'):>9}")
    print("-" * 52)
    print(f"ANN (target)            : {ann}")
    print(f"NF (torch deployed)     : {nf}")
    print(f"DEPLOYED full-test (SCM): {deployed}   <-- lossless gate")
    print(f"nevresim sim (subsample): {nevresim}")
    if ann and deployed:
        gap = ann - deployed
        print(f"LOSSLESS GAP (ANN-SCM)  : {gap:+.4f}  ({'LOSSLESS' if gap <= 0.006 else 'LOSSY'} @ ~1SE=0.005)")
    print(f"FAST budget (tuning)    : {fast_wall:.1f}s  ({'OK' if fast_wall < 120 else 'OVER'} <120s)")
    print(f"total pipeline wall     : {total_wall:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--dir":
        summarize(sys.argv[2])
    elif len(sys.argv) >= 2:
        summarize(f"generated/{sys.argv[1]}_phased_deployment_run")
    else:
        print(__doc__)
