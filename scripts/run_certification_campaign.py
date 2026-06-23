"""Frontier Phase 2 — the certification campaign (freeze floor → certify Fix B).

Runs the 9-cell mmixcore matrix end-to-end and drives the E6 certification
protocol: FREEZE each cell's current (controller) deployed metric as the
regression floor, then run the proven fast recipe (Fix B) per cell and CERTIFY
it against the frozen floor (deployed accuracy >= floor − eps AND wall <= budget).

This is the GPU run the protocol (docs/CERTIFICATION_PROTOCOL.md) says must
precede the first Fix-B flip. It only *measures + certifies*; landing a passing
flip (editing the template default + re-freezing at the new numbers) is a
reviewed commit done after reading this campaign's report.

It reuses the tested primitives:
  - mimarsinan.chip_simulation.cost_extraction._deployed_accuracy_from_run
  - mimarsinan.chip_simulation.certification.{freeze_cell,certify,...}

Artifacts written under docs/certification/:
  - regression_floor.json   the frozen per-cell floor book (E6 format)
  - campaign_results.json   full slow/fast measurements + verdicts (resumable)
  - campaign_report.md      human-readable summary

Resumable: a cell whose measurement already exists in campaign_results.json is
skipped unless --force. Each run's working dir is cleaned first so timing/caches
are not confounded by a prior run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from mimarsinan.chip_simulation.certification import (  # noqa: E402
    CertificationCell,
    CertificationFloorBook,
    certify,
    freeze_cell,
    load_floor_book,
    save_floor_book,
)
from mimarsinan.chip_simulation.cost_extraction import (  # noqa: E402
    _deployed_accuracy_from_run,
)

CERT_DIR = os.path.join(REPO, "docs", "certification")
FAST_CFG_DIR = os.path.join(REPO, "experiments", "certification")
FLOOR_BOOK = os.path.join(CERT_DIR, "regression_floor.json")
RESULTS = os.path.join(CERT_DIR, "campaign_results.json")
REPORT = os.path.join(CERT_DIR, "campaign_report.md")
GEN = os.path.join(REPO, "generated")


@dataclass
class Cell:
    """One matrix coordinate to freeze + certify."""

    name: str
    template: str
    firing: str
    sync: Optional[str]
    variant: str
    backend: str = "nevresim"
    fast_overrides: Dict[str, Any] = field(default_factory=dict)
    # Applied to BOTH the floor and candidate runs (env accommodations that do not
    # touch the deployed metric of record, e.g. disabling an unavailable backend).
    common_overrides: Dict[str, Any] = field(default_factory=dict)
    eps: float = 0.005
    wall_slack: float = 0.25

    def cert_cell(self) -> CertificationCell:
        return CertificationCell(self.firing, self.sync, self.backend, variant=self.variant)

    @property
    def has_fast_lever(self) -> bool:
        return bool(self.fast_overrides)


# Stabilize-steps buy back deployed accuracy at trivial wall cost (the wall budget
# is the ~28-min controller floor, so the fast recipe has enormous headroom — the
# binding constraint is accuracy, not speed). Calibration: LIF stab=400 deployed
# 0.962 vs controller floor 0.972, so the fast recipe is bumped to close the gap.
LIF_FAST = {"lif_blend_fast": True, "lif_blend_fast_stabilize_steps": 1200}
TTFS_FAST = {"ttfs_blend_fast": True, "ttfs_blend_fast_stabilize_steps": 1200}

CELLS: List[Cell] = [
    Cell("matrix_1_lif_rate",
         "templates/mnist_mmixcore_matrix_1_lif_rate.json",
         "lif", None, variant="rate", fast_overrides=dict(LIF_FAST)),
    Cell("matrix_2_lif_novena_offload_loihi",
         "templates/mnist_mmixcore_matrix_2_lif_novena_offload_loihi.json",
         "lif", None, variant="novena_offload", fast_overrides=dict(LIF_FAST),
         # lava/loihi sim is unavailable in this env (no env310); it is an auxiliary
         # backend, not the deployed metric of record (nevresim/SCM full-test), so
         # disabling it on both sides keeps the comparison fair + the metric intact.
         common_overrides={"enable_loihi_simulation": False}),
    Cell("matrix_3_lif_pruned_scheduled",
         "templates/mnist_mmixcore_matrix_3_lif_pruned_scheduled.json",
         "lif", None, variant="pruned_scheduled", fast_overrides=dict(LIF_FAST)),
    Cell("matrix_4_ttfs_analytical",
         "templates/mnist_mmixcore_matrix_4_ttfs_analytical.json",
         "ttfs", None, variant="analytical"),
    Cell("matrix_5_ttfs_quantized_offload",
         "templates/mnist_mmixcore_matrix_5_ttfs_quantized_offload.json",
         "ttfs_quantized", None, variant="offload"),
    Cell("matrix_6_ttfs_cycle_cascaded",
         "templates/mnist_mmixcore_matrix_6_ttfs_cycle_cascaded.json",
         "ttfs_cycle_based", "cascaded", variant="plain",
         fast_overrides=dict(TTFS_FAST), eps=0.01),
    Cell("matrix_7_ttfs_cycle_synchronized",
         "templates/mnist_mmixcore_matrix_7_ttfs_cycle_synchronized.json",
         "ttfs_cycle_based", "synchronized", variant="plain",
         fast_overrides=dict(TTFS_FAST)),
    Cell("matrix_8_ttfs_cycle_offload_scheduled_nobias",
         "templates/mnist_mmixcore_matrix_8_ttfs_cycle_offload_scheduled_nobias.json",
         "ttfs_cycle_based", "cascaded", variant="offload_scheduled_nobias",
         fast_overrides=dict(TTFS_FAST), eps=0.01),
    Cell("matrix_9_ttfs_vanilla_noWQ",
         "templates/mnist_mmixcore_matrix_9_ttfs_vanilla_noWQ.json",
         "ttfs", None, variant="vanilla_noWQ"),
]

# Escalation recipe for a cascaded cell whose plain fast recipe regresses below
# the accuracy floor: add the R2 STE-hedge (clean staircase-backward gradient).
R2_ESCALATION = {
    "ttfs_staircase_ste": True,
    "ttfs_staircase_ste_fast": True,
    "ttfs_ste_mix": 0.5,
}


def _load_results() -> Dict[str, Any]:
    if os.path.isfile(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_results(results: Dict[str, Any]) -> None:
    os.makedirs(CERT_DIR, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_config(base_config: Dict[str, Any], experiment_name: str,
                  overrides: Dict[str, Any]) -> str:
    cfg = copy.deepcopy(base_config)
    cfg["experiment_name"] = experiment_name
    cfg["deployment_parameters"].update(overrides)
    os.makedirs(FAST_CFG_DIR, exist_ok=True)
    path = os.path.join(FAST_CFG_DIR, experiment_name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return path


def _run_dir_for(experiment_name: str, pipeline_mode: str = "phased") -> str:
    # The pipeline names its working dir ``{name}_{pipeline_mode}_deployment_run``
    # (src/main.py); a template may declare a non-phased mode (e.g. matrix_9 is
    # ``vanilla``), so the run dir must use the config's actual pipeline_mode.
    return os.path.join(GEN, f"{experiment_name}_{pipeline_mode}_deployment_run")


def _measure(run_dir: str) -> Dict[str, Any]:
    acc = _deployed_accuracy_from_run(run_dir)
    steps_path = os.path.join(run_dir, "_GUI_STATE", "steps.json")
    steps = {}
    wall = 0.0
    if os.path.isfile(steps_path):
        with open(steps_path, "r", encoding="utf-8") as fh:
            raw = (json.load(fh) or {}).get("steps") or {}
        for name, st in raw.items():
            dur = (st.get("end_time", 0) or 0) - (st.get("start_time", 0) or 0)
            steps[name] = round(float(dur), 2)
            wall += float(dur)
    return {"deployed_accuracy": acc, "wall_clock_s": round(wall, 2), "steps": steps}


def _run(config_path: str, experiment_name: str, timeout: int,
         pipeline_mode: str = "phased") -> Dict[str, Any]:
    run_dir = _run_dir_for(experiment_name, pipeline_mode)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "run.py", "--headless", config_path],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and os.path.isfile(
        os.path.join(run_dir, "__target_metric.json"))
    measured = _measure(run_dir) if os.path.isdir(run_dir) else {}
    measured["proc_returncode"] = proc.returncode
    measured["proc_wall_s"] = round(elapsed, 1)
    measured["ok"] = ok
    if not ok:
        measured["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-25:])
    return measured


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def run_campaign(only: Optional[List[str]], phase: str, force: bool,
                 timeout: int) -> None:
    results = _load_results()
    commit = _commit()
    cells = [c for c in CELLS if (only is None or c.name in only)]
    do_freeze = phase in ("freeze", "all")
    do_certify = phase in ("certify", "all")

    for cell in cells:
        base = json.load(open(os.path.join(REPO, cell.template), encoding="utf-8"))
        pmode = base.get("pipeline_mode", "phased")
        slot = results.setdefault(cell.name, {"cell_key": cell.cert_cell().cell_key})

        # ---- FREEZE: the current (controller) deployed baseline (expensive,
        # recipe-independent — run once and cache). ----
        if do_freeze and (force or "floor" not in slot):
            exp = "cert_floor_" + cell.name
            cfg = _write_config(base, exp, dict(cell.common_overrides))
            print(f"[freeze] {cell.name} ...", flush=True)
            slot["floor"] = _run(cfg, exp, timeout, pmode)
            slot["floor"]["commit"] = commit
            _save_results(results)
            print(f"[freeze] {cell.name}: acc={slot['floor'].get('deployed_accuracy')} "
                  f"wall={slot['floor'].get('wall_clock_s')}s ok={slot['floor'].get('ok')}",
                  flush=True)

        # ---- CANDIDATE: the fast recipe (Fix B), or the floor itself (cheap —
        # re-run freely while tuning the recipe). ----
        if do_certify:
            if not cell.has_fast_lever:
                if slot.get("floor"):
                    slot["candidate"] = dict(slot["floor"])
                    slot["candidate"]["recipe"] = "no-fast-lever (already fast/lossless)"
                    _save_results(results)
            elif force or "candidate" not in slot:
                exp = "cert_fast_" + cell.name
                cfg = _write_config(
                    base, exp, {**cell.common_overrides, **cell.fast_overrides})
                print(f"[certify] {cell.name} (fast) ...", flush=True)
                slot["candidate"] = _run(cfg, exp, timeout, pmode)
                slot["candidate"]["recipe"] = dict(cell.fast_overrides)
                slot["candidate"]["commit"] = commit
                _save_results(results)
                print(f"[certify] {cell.name}: acc={slot['candidate'].get('deployed_accuracy')} "
                      f"wall={slot['candidate'].get('wall_clock_s')}s ok={slot['candidate'].get('ok')}",
                      flush=True)

    _freeze_and_certify(results, commit)
    _write_report(results)


def _freeze_and_certify(results: Dict[str, Any], commit: str) -> None:
    """Build the floor book from the floor runs, then certify every candidate."""
    book = CertificationFloorBook()
    for cell in CELLS:
        slot = results.get(cell.name)
        if not slot or not slot.get("floor", {}).get("ok"):
            continue
        f = slot["floor"]
        book = freeze_cell(
            book, cell.cert_cell(),
            deployed_accuracy=f["deployed_accuracy"],
            wall_clock_s=f["wall_clock_s"],
            eps=cell.eps, wall_clock_slack=cell.wall_slack,
            provenance={"commit": f.get("commit", commit), "samples": "full_test"},
        )
    save_floor_book(book, FLOOR_BOOK)

    for cell in CELLS:
        slot = results.get(cell.name)
        if not slot or not slot.get("candidate", {}).get("ok"):
            continue
        cand = slot["candidate"]
        verdict = certify(
            cell.cert_cell(),
            deployed_accuracy=cand["deployed_accuracy"],
            wall_clock_s=cand["wall_clock_s"],
            floor_book=book,
        )
        slot["verdict"] = {
            "status": verdict.status.value,
            "accuracy_ok": verdict.accuracy_ok,
            "wall_clock_ok": verdict.wall_clock_ok,
            "reason": verdict.reason,
        }
    _save_results(results)


def _write_report(results: Dict[str, Any]) -> None:
    lines = ["# Frontier Phase 2 — certification campaign report", "",
             "Per-cell freeze (controller floor) + Fix-B fast candidate + E6 verdict.",
             "", "| cell | floor acc | floor wall | fast acc | fast wall | speedup | verdict |",
             "|------|-----------|-----------|----------|-----------|---------|---------|"]
    for cell in CELLS:
        slot = results.get(cell.name, {})
        f, c = slot.get("floor", {}), slot.get("candidate", {})
        v = slot.get("verdict", {})
        fa = f.get("deployed_accuracy"); fw = f.get("wall_clock_s")
        ca = c.get("deployed_accuracy"); cw = c.get("wall_clock_s")
        spd = f"{fw/cw:.1f}x" if (fw and cw) else "-"
        lines.append(
            f"| {cell.cert_cell().cell_key} | "
            f"{fa} | {fw} | {ca} | {cw} | {spd} | {v.get('status','-')} |")
    lines += ["", f"Floor book: `{os.path.relpath(FLOOR_BOOK, REPO)}`",
              f"Results: `{os.path.relpath(RESULTS, REPO)}`", ""]
    for cell in CELLS:
        v = results.get(cell.name, {}).get("verdict")
        if v:
            lines.append(f"- **{cell.name}**: {v['status']} — {v['reason']}")
    os.makedirs(CERT_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None, help="cell names to run")
    p.add_argument("--phase", choices=["freeze", "certify", "all"], default="all",
                   help="freeze=floors only; certify=fast candidates only; all=both")
    p.add_argument("--force", action="store_true", help="re-run cells already measured")
    p.add_argument("--timeout", type=int, default=2400, help="per-run timeout (s)")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild floor book + verdicts + report from existing results")
    args = p.parse_args(argv)

    if args.report_only:
        results = _load_results()
        _freeze_and_certify(results, _commit())
        _write_report(results)
        return 0

    run_campaign(args.only, args.phase, args.force, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
