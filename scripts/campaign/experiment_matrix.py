"""The publication experiment matrix: backlog batch generators + ledger aggregator.

No GPU is touched here. This module GENERATES backlog-schema batch dicts (the
same shape ``scheduler.instantiate`` / ``Scheduler.refill`` consume — see
``runs/campaign/backlog.json``) for three publication studies, and AGGREGATES
the resulting ``runs/campaign/ledger.jsonl`` rows into the study tables:

  F1  CIs + ablations   — multi-seed cells -> mean +/- 95% CI per cell.
  F2  baseline head-to-head — percentile-norm (quantile 1.0) vs the default
      activation-scale clip (0.99) on covered/valid cells -> per-cell delta_pp.
  F3  dual-regime        — from_scratch vs pretrained per (model,dataset)
      -> per-(model,dataset) delta_pp.

The generated batches are ENQUEUED later by the scheduler; the aggregator is run
after those runs finalize and append per-seed ledger rows. The aggregator also
writes a markdown findings report under ``docs/research/findings/``.

CLI:
  generate  python scripts/campaign/experiment_matrix.py generate \
                --study {f1,f2,f3,all} [--out runs/campaign/backlog_F.json] [--append]
  aggregate python scripts/campaign/experiment_matrix.py aggregate \
                [--ledger runs/campaign/ledger.jsonl] [--out docs/.../F_matrix.md]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import string
import time
from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN_DIR = os.path.join(REPO, "runs", "campaign")
LEDGER = os.path.join(CAMPAIGN_DIR, "ledger.jsonl")
FINDINGS_DIR = os.path.join(REPO, "docs", "research", "findings")

# F2 baseline arms: the per-perceptron decode-scale quantile. 0.99 is the
# historical default activation-scale clip; 1.0 is percentile-norm (no clip).
DEFAULT_ACTIVATION_SCALE_QUANTILE = 0.99
PERCENTILE_NORM_QUANTILE = 1.0
QUANTILE_KEY = "deployment_parameters.activation_scale_quantile"

# F3 regime arm: preload ImageNet/host weights (pretrained) vs from_scratch.
PRELOAD_KEY = "deployment_parameters.preload_weights"

# Student-t two-sided 95% critical values indexed by degrees of freedom
# (df = n-1). Index 0 (df=0) is unused; for df >= 30 we use the normal
# approximation 1.96. Self-contained so the CI math needs no scipy.
T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045,
}
_T_CRIT_NORMAL = 1.96


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return 0.0
    return T_CRIT_95.get(df, _T_CRIT_NORMAL)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class BatchSchemaError(ValueError):
    """A batch dict that ``scheduler.instantiate`` could not consume."""


def validate_batch(batch: dict) -> None:
    """Raise ``BatchSchemaError`` unless ``batch`` is a valid backlog batch.

    The contract is exactly what ``Scheduler.refill`` + ``instantiate`` read:
    an ``id``, a ``template`` path, a non-empty ``grid`` (each axis a non-empty
    list), and an ``id_template`` whose ``{field}`` placeholders all resolve to
    a grid-axis leaf name. ``base`` (when present) is a flat dotted->value map.
    """
    if not isinstance(batch, dict):
        raise BatchSchemaError("batch must be a dict")
    for key in ("id", "template", "grid", "id_template"):
        if key not in batch:
            raise BatchSchemaError(f"missing required key {key!r}")
    if not (isinstance(batch["id"], str) and batch["id"]):
        raise BatchSchemaError("id must be a non-empty string")
    if not (isinstance(batch["template"], str) and batch["template"]):
        raise BatchSchemaError("template must be a non-empty string")
    if not isinstance(batch["id_template"], str):
        raise BatchSchemaError("id_template must be a string")
    grid = batch["grid"]
    if not (isinstance(grid, dict) and grid):
        raise BatchSchemaError("grid must be a non-empty dict")
    leaves = set()
    for path, vals in grid.items():
        if not isinstance(path, str) or not path:
            raise BatchSchemaError(f"grid key must be a non-empty string: {path!r}")
        if not (isinstance(vals, list) and len(vals) >= 1):
            raise BatchSchemaError(f"grid axis {path!r} must be a non-empty list")
        leaves.add(path.split(".")[-1])
    fields = {fn for _, fn, _, _ in string.Formatter().parse(batch["id_template"]) if fn}
    missing = fields - leaves
    if missing:
        raise BatchSchemaError(
            f"id_template references {sorted(missing)} not in grid leaves {sorted(leaves)}")
    base = batch.get("base", {})
    if base is not None and not isinstance(base, dict):
        raise BatchSchemaError("base must be a dict (dotted->value)")
    for k in (base or {}):
        if not (isinstance(k, str) and k):
            raise BatchSchemaError(f"base key must be a non-empty dotted string: {k!r}")


# ---------------------------------------------------------------------------
# Matrix cells
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixCell:
    """One honestly-priced, covered/valid (model, dataset) point of the matrix."""

    model: str
    dataset: str
    template: str
    schedules: Tuple[str, ...] = ("cascaded", "synchronized")
    depths: Tuple[int, ...] = ()
    base: Tuple[Tuple[str, object], ...] = ()
    tags: Tuple[Tuple[str, object], ...] = ()

    def base_dict(self) -> Dict[str, object]:
        return dict(self.base)

    def tag_dict(self) -> Dict[str, object]:
        return {"model": self.model, "dataset": self.dataset, **dict(self.tags)}

    def slug(self) -> str:
        return f"{self.model}_{_short_dataset(self.dataset)}"


def _short_dataset(name: str) -> str:
    return name.replace("_DataProvider", "").lower()


def _grid(cell: MatrixCell, seeds: Sequence[int],
          extra: Optional[Dict[str, list]] = None) -> Dict[str, list]:
    grid: Dict[str, list] = {"data_provider_name": [cell.dataset]}
    if cell.schedules:
        grid["deployment_parameters.ttfs_cycle_schedule"] = list(cell.schedules)
    if cell.depths:
        grid["deployment_parameters.model_config.depth"] = list(cell.depths)
    if extra:
        grid.update(extra)
    grid["seed"] = list(seeds)
    return grid


def _id_template(prefix: str, grid: Dict[str, list]) -> str:
    """A ``{leaf}`` id_template covering every multi-valued grid axis."""
    parts = [prefix]
    for path, vals in grid.items():
        leaf = path.split(".")[-1]
        if leaf == "data_provider_name":
            parts.append("{data_provider_name}")
        elif leaf == "ttfs_cycle_schedule":
            parts.append("{ttfs_cycle_schedule}")
        elif leaf == "depth":
            parts.append("d{depth}")
        elif leaf == "seed":
            parts.append("s{seed}")
        else:
            parts.append(leaf + "_{" + leaf + "}")
    return "_".join(parts)


def _batch(study: str, suffix: str, cell: MatrixCell, grid: Dict[str, list],
           *, priority: int, extra_base: Optional[Dict[str, object]] = None,
           extra_tags: Optional[Dict[str, object]] = None) -> dict:
    bid = f"{study.lower()}_{cell.slug()}_{suffix}"
    base = cell.base_dict()
    if extra_base:
        base.update(extra_base)
    tags = {"study": study, **cell.tag_dict()}
    if extra_tags:
        tags.update(extra_tags)
    batch = {
        "id": bid,
        "template": cell.template,
        "base": base,
        "grid": grid,
        "id_template": _id_template(f"{study.lower()}_{cell.slug()}_{suffix}", grid),
        "priority": priority,
        "tags": tags,
        "enabled": False,
    }
    validate_batch(batch)
    return batch


# ---------------------------------------------------------------------------
# F1 — CIs + ablations
# ---------------------------------------------------------------------------
def gen_f1_batches(cells: Iterable[MatrixCell], *,
                   seeds: Sequence[int] = (0, 1, 2, 3, 4),
                   priority: int = 35) -> List[dict]:
    """Multi-seed cells across the honestly-priced matrix for mean +/- CI."""
    if len(seeds) < 2:
        raise ValueError("F1 needs >= 2 seeds for a confidence interval")
    out = []
    for cell in cells:
        grid = _grid(cell, seeds)
        out.append(_batch("F1", "ci", cell, grid, priority=priority,
                          extra_tags={"kind": "ci_ablation"}))
    return out


# ---------------------------------------------------------------------------
# F2 — baseline head-to-head (percentile-norm vs default activation-scale)
# ---------------------------------------------------------------------------
def gen_f2_batches(cells: Iterable[MatrixCell], *,
                   seeds: Sequence[int] = (0, 1, 2),
                   priority: int = 40) -> List[dict]:
    """Both baseline arms (quantile 1.0 vs 0.99) on each covered/valid cell."""
    out = []
    arms = [DEFAULT_ACTIVATION_SCALE_QUANTILE, PERCENTILE_NORM_QUANTILE]
    for cell in cells:
        grid = _grid(cell, seeds, extra={QUANTILE_KEY: arms})
        out.append(_batch("F2", "baseline", cell, grid, priority=priority,
                          extra_tags={"kind": "baseline_head_to_head"}))
    return out


# ---------------------------------------------------------------------------
# F3 — dual-regime (from_scratch vs pretrained)
# ---------------------------------------------------------------------------
def gen_f3_batches(cells: Iterable[MatrixCell], *,
                   seeds: Sequence[int] = (0, 1, 2),
                   priority: int = 40) -> List[dict]:
    """from_scratch vs pretrained per (model, dataset)."""
    out = []
    for cell in cells:
        grid = _grid(cell, seeds, extra={PRELOAD_KEY: [False, True]})
        out.append(_batch("F3", "dualregime", cell, grid, priority=priority,
                          extra_tags={"kind": "dual_regime"}))
    return out


def gen_all_batches(cells: Iterable[MatrixCell], **kw) -> List[dict]:
    cells = list(cells)
    return gen_f1_batches(cells, **_kw(kw, "f1")) + \
        gen_f2_batches(cells, **_kw(kw, "f2")) + \
        gen_f3_batches(cells, **_kw(kw, "f3"))


def _kw(kw: dict, _study: str) -> dict:
    return {k: v for k, v in kw.items() if k in ("seeds", "priority")}


# ---------------------------------------------------------------------------
# Ledger reading
# ---------------------------------------------------------------------------
def read_ledger(path: str = LEDGER) -> List[dict]:
    """Tolerant JSONL reader: skip blank lines and un-parseable rows."""
    rows: List[dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _metric(row: dict) -> Optional[float]:
    v = row.get("deployed_acc")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """(mean, 95% t-CI half-width). Half-width is 0 for n < 2."""
    n = len(values)
    mean = fmean(values)
    if n < 2:
        return mean, 0.0
    se = stdev(values) / math.sqrt(n)
    return mean, _t_crit_95(n - 1) * se


# ---------------------------------------------------------------------------
# F1 aggregator — per-cell mean +/- CI
# ---------------------------------------------------------------------------
F1_CELL_KEYS = ("model", "dataset", "schedule", "depth")


def aggregate_f1(rows: Iterable[dict]) -> List[dict]:
    """Per-cell mean +/- 95% CI over the F1 multi-seed rows."""
    groups: Dict[tuple, List[Tuple[float, str]]] = {}
    for r in rows:
        if r.get("study") != "F1":
            continue
        m = _metric(r)
        if m is None:
            continue
        key = tuple(r.get(k) for k in F1_CELL_KEYS)
        groups.setdefault(key, []).append((m, str(r.get("run_id", ""))))
    table = []
    for key, items in sorted(groups.items(), key=_cell_sort_key):
        vals = [v for v, _ in items]
        mean, ci = _mean_ci(vals)
        table.append({
            "cell": list(key),
            "n_seeds": len(vals),
            "deployed_acc_mean": mean,
            "ci95": ci,
            "run_ids": sorted(rid for _, rid in items if rid),
        })
    return table


# ---------------------------------------------------------------------------
# F2 aggregator — head-to-head delta (percentile-norm - default)
# ---------------------------------------------------------------------------
def aggregate_f2(rows: Iterable[dict]) -> List[dict]:
    """Per-cell delta_pp = percentile-norm mean - default mean (only paired cells)."""
    default: Dict[tuple, List[float]] = {}
    percentile: Dict[tuple, List[float]] = {}
    for r in rows:
        if r.get("study") != "F2":
            continue
        m = _metric(r)
        if m is None:
            continue
        q = r.get("activation_scale_quantile")
        key = tuple(r.get(k) for k in F1_CELL_KEYS)
        if _is_close(q, DEFAULT_ACTIVATION_SCALE_QUANTILE):
            default.setdefault(key, []).append(m)
        elif _is_close(q, PERCENTILE_NORM_QUANTILE):
            percentile.setdefault(key, []).append(m)
    table = []
    for key in sorted(set(default) & set(percentile), key=lambda k: (str(k),)):
        d_mean, d_ci = _mean_ci(default[key])
        p_mean, p_ci = _mean_ci(percentile[key])
        table.append({
            "cell": list(key),
            "default_mean": d_mean,
            "default_ci95": d_ci,
            "n_default": len(default[key]),
            "percentile_mean": p_mean,
            "percentile_ci95": p_ci,
            "n_percentile": len(percentile[key]),
            "delta_pp": (p_mean - d_mean) * 100.0,
        })
    return table


# ---------------------------------------------------------------------------
# F3 aggregator — dual-regime delta (pretrained - from_scratch)
# ---------------------------------------------------------------------------
F3_CELL_KEYS = ("model", "dataset")


def aggregate_f3(rows: Iterable[dict]) -> List[dict]:
    """Per-(model,dataset) delta_pp = pretrained mean - from_scratch mean."""
    scratch: Dict[tuple, List[float]] = {}
    pretrained: Dict[tuple, List[float]] = {}
    for r in rows:
        if r.get("study") != "F3":
            continue
        m = _metric(r)
        if m is None:
            continue
        key = tuple(r.get(k) for k in F3_CELL_KEYS)
        if bool(r.get("preload_weights")):
            pretrained.setdefault(key, []).append(m)
        else:
            scratch.setdefault(key, []).append(m)
    table = []
    for key in sorted(set(scratch) & set(pretrained), key=lambda k: (str(k),)):
        s_mean, s_ci = _mean_ci(scratch[key])
        p_mean, p_ci = _mean_ci(pretrained[key])
        table.append({
            "cell": list(key),
            "from_scratch_mean": s_mean,
            "from_scratch_ci95": s_ci,
            "n_from_scratch": len(scratch[key]),
            "pretrained_mean": p_mean,
            "pretrained_ci95": p_ci,
            "n_pretrained": len(pretrained[key]),
            "delta_pp": (p_mean - s_mean) * 100.0,
        })
    return table


def _is_close(a, b, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _cell_sort_key(item):
    key, _ = item
    return tuple("" if k is None else str(k) for k in key)


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------
def _fmt_cell(cell: Sequence) -> str:
    return " / ".join("" if c is None else str(c) for c in cell)


def _f1_md(table: List[dict]) -> str:
    lines = ["## F1 — confidence intervals (per cell, mean +/- 95% CI)", "",
             "| cell (model/dataset/schedule/depth) | n | deployed mean | +/- 95% CI |",
             "|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {r['n_seeds']} | "
            f"{r['deployed_acc_mean']:.4f} | {r['ci95']:.4f} |")
    return "\n".join(lines)


def _f2_md(table: List[dict]) -> str:
    lines = ["## F2 — baseline head-to-head (percentile-norm vs default activation-scale)",
             "",
             "| cell | default mean | percentile mean | delta (pp) |",
             "|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {r['default_mean']:.4f} | "
            f"{r['percentile_mean']:.4f} | {r['delta_pp']:+.2f} |")
    return "\n".join(lines)


def _f3_md(table: List[dict]) -> str:
    lines = ["## F3 — dual-regime (from_scratch vs pretrained)", "",
             "| model / dataset | from_scratch mean | pretrained mean | delta (pp) |",
             "|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {r['from_scratch_mean']:.4f} | "
            f"{r['pretrained_mean']:.4f} | {r['delta_pp']:+.2f} |")
    return "\n".join(lines)


def render_findings_markdown(*, f1=None, f2=None, f3=None) -> str:
    blocks = ["# Publication experiment matrix — F1/F2/F3 findings", "",
              f"_generated {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC_", ""]
    blocks.append(_f1_md(f1 or []))
    blocks.append("")
    blocks.append(_f2_md(f2 or []))
    blocks.append("")
    blocks.append(_f3_md(f3 or []))
    blocks.append("")
    return "\n".join(blocks)


def write_findings_markdown(path: Optional[str], *, f1=None, f2=None, f3=None) -> str:
    """Write the F-matrix findings markdown; default lands in ``FINDINGS_DIR``."""
    if path is None:
        path = os.path.join(FINDINGS_DIR, "F_experiment_matrix.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(render_findings_markdown(f1=f1, f2=f2, f3=f3))
    return path


# ---------------------------------------------------------------------------
# Default publication matrix — covered/valid cells only (deep_cnn/lenet5 are
# the VALID trainable-deep / shallow vehicles per the on-chip-majority gate).
# ---------------------------------------------------------------------------
def default_matrix() -> List[MatrixCell]:
    return [
        MatrixCell(model="deep_cnn", dataset="MNIST_DataProvider",
                   template="templates/mnist_deep_cnn_d8_cascaded.json",
                   schedules=("cascaded", "synchronized"), depths=(4, 6, 8),
                   tags=(("ws", "F"), ("trainable", True))),
        MatrixCell(model="deep_cnn", dataset="FashionMNIST_DataProvider",
                   template="templates/mnist_deep_cnn_d8_cascaded.json",
                   schedules=("cascaded", "synchronized"), depths=(4, 6, 8),
                   tags=(("ws", "F"), ("trainable", True))),
        MatrixCell(model="lenet5", dataset="MNIST_DataProvider",
                   template="templates/mnist_lenet5_synchronized.json",
                   schedules=("synchronized",),
                   tags=(("ws", "F"),)),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_generate(args) -> int:
    cells = default_matrix()
    study = args.study.lower()
    if study == "f1":
        batches = gen_f1_batches(cells)
    elif study == "f2":
        batches = gen_f2_batches(cells)
    elif study == "f3":
        batches = gen_f3_batches(cells)
    elif study == "all":
        batches = gen_all_batches(cells)
    else:
        raise SystemExit(f"unknown study {args.study!r}")
    out = args.out or os.path.join(CAMPAIGN_DIR, f"backlog_{study}.json")
    existing = []
    if args.append and os.path.isfile(out):
        try:
            existing = json.load(open(out))
        except (OSError, ValueError):
            existing = []
    have = {b.get("id") for b in existing}
    merged = existing + [b for b in batches if b["id"] not in have]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(json.dumps({"study": study, "batches": len(batches),
                      "written": len(merged), "out": out}))
    return 0


def cmd_aggregate(args) -> int:
    rows = read_ledger(args.ledger)
    f1 = aggregate_f1(rows)
    f2 = aggregate_f2(rows)
    f3 = aggregate_f3(rows)
    path = write_findings_markdown(args.out, f1=f1, f2=f2, f3=f3)
    print(json.dumps({"f1_cells": len(f1), "f2_cells": len(f2),
                      "f3_cells": len(f3), "report": path}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--study", default="all", choices=["f1", "f2", "f3", "all"])
    g.add_argument("--out", default=None)
    g.add_argument("--append", action="store_true")
    g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("aggregate")
    a.add_argument("--ledger", default=LEDGER)
    a.add_argument("--out", default=None)
    a.set_defaults(fn=cmd_aggregate)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
