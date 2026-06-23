"""Generate the Lane-2 (GPU data-collection) config variants + dispatch manifest.

Pure data collection on current `main` (independent of the Lane-1 code work):
  B1 — S-sweep for AC2/AC3: each cell at S in {4,8,16,32}, seed 0 (AC3 monotonicity);
       + 3 seeds at S=32 on the lossless-candidate cells (AC2 seed-σ). sanafe off
       (only the deployed SCM full-test accuracy is needed) -> faster.
  B4 — AC6 backend coverage: each cell once with SANA-FE on (zero-crash through
       SCM/HCM/nevresim/SANA-FE). loihi/lava is unavailable in this env -> N/A.

Writes configs under experiments/finish/ (gitignored) + a dispatch manifest. The
dispatcher (gpu_dispatch.py) runs `python run.py --headless <cfg>`; verdicts are
computed later from each run's __target_metric.json + steps.json.
"""
from __future__ import annotations

import copy
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "experiments", "finish")
TEMPLATES = sorted(glob.glob(os.path.join(REPO, "templates", "mnist_mmixcore_matrix_*.json")))

S_GRID = [4, 8, 16, 32]
SIGMA_SEEDS = [0, 1, 2]
# cells whose lossless claim is worth a seed-σ band (skip the cascaded death-cascade ones)
SIGMA_CELLS = {"matrix_1", "matrix_4", "matrix_5", "matrix_7", "matrix_9"}


def _cell(template_path: str) -> str:
    base = os.path.basename(template_path)
    # mnist_mmixcore_matrix_3_lif_pruned_scheduled.json -> matrix_3
    parts = base.split("_")
    return "_".join(parts[2:4])  # matrix_<n>


def _write(cfg: dict, name: str) -> str:
    cfg = copy.deepcopy(cfg)
    cfg["experiment_name"] = name
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return path


def _set_S(cfg: dict, S: int) -> None:
    pc = cfg["platform_constraints"]
    pc["simulation_steps"] = S
    pc["target_tq"] = S


def _job(cfg_path: str, jid: str, need_mb: int = 8000) -> dict:
    return {"id": jid, "mode": "fit", "need_mb": need_mb,
            "cmd": ["python", "run.py", "--headless", cfg_path], "cwd": REPO}


def build():
    jobs = []
    for tpl in TEMPLATES:
        base = json.load(open(tpl))
        cell = _cell(tpl)

        # B1 — S-sweep, seed 0 (AC3 monotonicity curve)
        for S in S_GRID:
            cfg = copy.deepcopy(base)
            cfg["seed"] = 0
            _set_S(cfg, S)
            cfg["deployment_parameters"]["enable_sanafe_simulation"] = False
            cfg["deployment_parameters"]["enable_loihi_simulation"] = False
            name = f"finish_b1_{cell}_S{S}_s0"
            jobs.append(_job(_write(cfg, name), name))

        # B1 — seed-σ at S=32 for lossless-candidate cells
        if cell in SIGMA_CELLS:
            for seed in SIGMA_SEEDS[1:]:  # seed 0 already covered above
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                _set_S(cfg, 32)
                cfg["deployment_parameters"]["enable_sanafe_simulation"] = False
                cfg["deployment_parameters"]["enable_loihi_simulation"] = False
                name = f"finish_b1_{cell}_S32_s{seed}"
                jobs.append(_job(_write(cfg, name), name))

        # B4 — AC6 backend coverage (SANA-FE on), shipped S, seed 0
        cfg = copy.deepcopy(base)
        cfg["seed"] = 0
        cfg["deployment_parameters"]["enable_sanafe_simulation"] = True
        cfg["deployment_parameters"]["enable_loihi_simulation"] = False
        name = f"finish_b4_{cell}_sanafe"
        jobs.append(_job(_write(cfg, name), name))

    manifest = os.path.join(OUTDIR, "manifest_b1_b4.json")
    with open(manifest, "w") as fh:
        json.dump(jobs, fh, indent=2)
    return jobs, manifest


if __name__ == "__main__":
    jobs, manifest = build()
    b1 = [j for j in jobs if "_b1_" in j["id"]]
    b4 = [j for j in jobs if "_b4_" in j["id"]]
    print(f"generated {len(jobs)} jobs: B1(S-sweep+σ)={len(b1)}, B4(backend)={len(b4)}")
    print(f"manifest: {manifest}")
