"""Phase-A audit: extract config + per-step outcome for the 9 runs.

Run from project root: `python experiments/audit_runs.py`
"""
import glob
import json
import os

RUN_GLOB = "generated/*_20260619_12*"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


def _last_metric(step):
    ms = step.get("metrics") or []
    if not ms:
        return None
    last = ms[-1]
    if isinstance(last, dict):
        for k in ("value", "metric", "accuracy", "acc"):
            if k in last:
                return last[k]
        return last
    return last


def _config_point(cfg):
    """Pull the §3 config axes out of a run config (best-effort across schema)."""
    dp = cfg.get("deployment_parameters", cfg)
    pick = {}
    for key in (
        "firing_mode", "spike_mode", "simulation_mode", "sync_mode",
        "synchronized", "offload_encoding", "encoding", "thresholding_mode",
        "spike_generation_mode", "target_tq", "weight_bits", "from_scratch",
        "pruning", "mapping_strategy",
    ):
        if key in dp:
            pick[key] = dp[key]
    pc = dp.get("platform_constraints") or cfg.get("platform_constraints") or {}
    for key in ("allow_coalescing", "allow_neuron_splitting", "allow_scheduling", "target_tq", "weight_bits"):
        if key in pc:
            pick[key] = pc[key]
    return pick


def main():
    runs = sorted(glob.glob(RUN_GLOB))
    print(f"# Phase-A audit — {len(runs)} runs\n")
    summary = []
    for d in runs:
        name = os.path.basename(d)
        gui = os.path.join(d, "_GUI_STATE")
        cfg = _load(os.path.join(d, "_RUN_CONFIG", "config.json"))
        info = _load(os.path.join(gui, "run_info.json"))
        steps_doc = _load(os.path.join(gui, "steps.json"))
        steps = steps_doc.get("steps", {}) if isinstance(steps_doc, dict) else {}

        exp = info.get("config_summary", {}).get("experiment_name", name)
        status = info.get("status")
        error = info.get("error")
        print(f"## {exp}")
        print(f"dir: {name}")
        print(f"run status: {status}   error: {error!r}")
        print(f"config: {json.dumps(_config_point(cfg))}")
        ordered = info.get("step_names") or list(steps.keys())
        failed_step = None
        for sname in ordered:
            st = steps.get(sname)
            if not st:
                print(f"  - {sname:32s} | (not reached)")
                continue
            sstat = st.get("status")
            dur = None
            if st.get("start_time") and st.get("end_time"):
                dur = round(st["end_time"] - st["start_time"], 1)
            metric = _last_metric(st)
            print(f"  - {sname:32s} | {str(sstat):10s} | metric={metric} | {dur}s")
            if sstat and sstat not in ("completed", "skipped"):
                failed_step = failed_step or sname
        summary.append((exp, status, failed_step, error))
        print()

    print("\n# ===== one-line summary =====")
    for exp, status, failed, error in summary:
        print(f"{exp:48s} | {str(status):10s} | failed_at={failed} | err={error!r}")


if __name__ == "__main__":
    main()
