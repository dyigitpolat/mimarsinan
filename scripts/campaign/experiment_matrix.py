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
  B2  CIFAR breadth      — extend the dataset-margin law to RGB: deep_cnn /
      lenet5 (both VALID at 3x32x32) on CIFAR10 + CIFAR100, multi-seed.

The generated batches are ENQUEUED later by the scheduler; the aggregator is run
after those runs finalize. The aggregator reads TWO ledger schemas: (1) the
study-tagged per-seed rows (``study:F1|F2|F3`` + flat ``deployed_acc``) the
generators target, and (2) the PRE-AGGREGATED SUMMARY rows the live campaign
actually writes — one row per cell, run_ids prefixed ``f1_``/``f2_``/``f3_``,
both schedule arms in one row (``{schedule}_deployed_mean``/``_std_pp``). Summary
cells carry the row's ``std_pp`` honestly with the 95% CI marked UNAVAILABLE (a
std is never reshaped into a CI) and emit crashed/0-seed arms as explicit
NOT-MEASURED cells. The aggregator also writes a markdown findings report under
``docs/research/findings/``.

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
# The pretrained source the ``preload_weights=True`` arm derives (mirrors
# ``DeploymentPlan.PRETRAINED_WEIGHT_SOURCE``); a cell's ``pretrained_source`` must
# equal this to be couplable to the boolean grid (a checkpoint path cannot).
PRETRAINED_WEIGHT_SOURCE = "torchvision"

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
    """One honestly-priced, covered/valid (model, dataset) point of the matrix.

    ``pretrained_source`` marks a vehicle that HAS a real pretrained source — a
    torchvision ImageNet factory (``"torchvision"``) or a checkpoint path (e.g.
    ``"runs/imagenet/resnet50_state.pt"``). Only such cells get an F3 pretrained
    arm; a native from-scratch vehicle (``None``) has no pretrained source, so an
    F3 pretrained arm for it would be ill-posed (there is no pretrained deep_cnn).
    """

    model: str
    dataset: str
    template: str
    schedules: Tuple[str, ...] = ("cascaded", "synchronized")
    depths: Tuple[int, ...] = ()
    base: Tuple[Tuple[str, object], ...] = ()
    tags: Tuple[Tuple[str, object], ...] = ()
    pretrained_source: Optional[str] = None

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
    """from_scratch vs pretrained per (model, dataset).

    The pretrained arm (``preload_weights=True``) is emitted ONLY for cells whose
    vehicle HAS a real pretrained source (``cell.pretrained_source``); for a native
    from-scratch vehicle (no pretrained source) emitting it would be ill-posed —
    ``weight_source='torchvision'`` has no factory to resolve, so the run crashes
    rc=1.

    The arms are selected by the ``preload_weights`` boolean grid axis, which keeps
    the two arms COUPLED in a single cartesian batch (a grid cannot express a
    per-arm ``weight_source`` override): ``True`` derives ``weight_source=
    'torchvision'`` (the pretrained factory), ``False`` leaves it unset =>
    from_scratch (unchanged). A torchvision-factory source is therefore the natural
    dual-regime vehicle; a checkpoint-only source cannot be derived from the boolean
    and is rejected here so a mis-marked cell fails loudly instead of silently
    emitting a from_scratch-vs-from_scratch pair.

    Cells with no pretrained source are SKIPPED (no spurious dual-regime row).
    """
    out = []
    for cell in cells:
        if cell.pretrained_source is None:
            continue  # no pretrained source => no computable dual-regime contrast
        if cell.pretrained_source != PRETRAINED_WEIGHT_SOURCE:
            raise ValueError(
                f"F3 dual-regime cell {cell.slug()!r} has pretrained_source="
                f"{cell.pretrained_source!r}; only the torchvision factory source "
                f"({PRETRAINED_WEIGHT_SOURCE!r}) can be coupled to the preload_weights "
                f"boolean grid. A checkpoint source needs a per-arm weight_source "
                f"override (two batches), not this generator."
            )
        grid = _grid(cell, seeds, extra={PRELOAD_KEY: [False, True]})
        out.append(_batch("F3", "dualregime", cell, grid, priority=priority,
                          extra_tags={"kind": "dual_regime"}))
    return out


def gen_all_batches(cells: Iterable[MatrixCell], **kw) -> List[dict]:
    """F1/F2 over the supplied (from-scratch) cells; F3 over the dual-regime cells.

    F3's from_scratch-vs-pretrained contrast needs a vehicle with a real pretrained
    source, so it runs on ``f3_dual_regime_matrix()`` (not the native default cells,
    which have no pretrained source and would be skipped).
    """
    cells = list(cells)
    return gen_f1_batches(cells, **_kw(kw, "f1")) + \
        gen_f2_batches(cells, **_kw(kw, "f2")) + \
        gen_f3_batches(f3_dual_regime_matrix(), **_kw(kw, "f3"))


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
# Two ledger schemas the aggregators consume
# ---------------------------------------------------------------------------
# (1) STUDY-TAGGED per-seed rows: ``study:F1|F2|F3`` + a flat ``deployed_acc``
#     (one row per seed). This is the generated batches' intended write-back.
# (2) PRE-AGGREGATED SUMMARY rows: what the LIVE campaign actually writes — ONE
#     row per cell, run_ids prefixed ``f1_``/``f2_``/``f3_``, both schedule arms
#     in one row (``{schedule}_deployed_mean``/``{schedule}_deployed_std_pp``),
#     carrying ``n_seeds``/``ann_test_acc_mean``/validity/verdict. A summary row
#     has NO per-seed sample, so we carry its ``std_pp`` honestly and mark the CI
#     UNAVAILABLE — we never reconstruct a 95% CI from a std.
_F_SCHEDULES = ("cascaded", "synchronized")
_RUN_ID_LIST_KEYS = ("cascaded_run_ids", "synchronized_run_ids", "run_ids",
                     "failed_run_ids", "failed_run_ids_rc1")


def _summary_study(row: dict) -> Optional[str]:
    """``F1``/``F2``/``F3`` if a SUMMARY row's item_id or any run_id is so
    prefixed, else ``None``. The live ledger's only F-study membership signal."""
    tags = [str(row.get("item_id", ""))]
    for k in _RUN_ID_LIST_KEYS:
        v = row.get(k)
        if isinstance(v, list):
            tags.extend(str(x) for x in v)
    for study in ("F1", "F2", "F3"):
        if any(t.startswith(study.lower() + "_") for t in tags):
            return study
    return None


def _f_number(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# F1 aggregator — per-cell mean +/- CI
# ---------------------------------------------------------------------------
F1_CELL_KEYS = ("model", "dataset", "schedule", "depth")


def aggregate_f1(rows: Iterable[dict]) -> List[dict]:
    """Per-cell deployed mean over F1 cells, from BOTH ledger schemas.

    Study-tagged per-seed rows -> mean +/- 95% t-CI (``source:per_seed``).
    Live summary rows -> one cell per (row, schedule arm) carrying the row's
    mean + ``std_pp`` with the CI marked unavailable (``source:summary``); a
    crashed arm (no mean) is emitted as an explicit NOT-MEASURED cell.
    """
    rows = list(rows)
    table = _aggregate_f1_per_seed(rows)
    table.extend(_aggregate_f1_summary(rows))
    return table


def _aggregate_f1_per_seed(rows: Sequence[dict]) -> List[dict]:
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
            "source": "per_seed",
            "cell": list(key),
            "n_seeds": len(vals),
            "deployed_acc_mean": mean,
            "std_pp": None,
            "ci95": ci,
            "ci_available": True,
            "measured": True,
            "run_ids": sorted(rid for _, rid in items if rid),
        })
    return table


def _aggregate_f1_summary(rows: Sequence[dict]) -> List[dict]:
    table = []
    for r in rows:
        if _summary_study(r) != "F1":
            continue
        for schedule in _F_SCHEDULES:
            mean = _f_number(r, f"{schedule}_deployed_mean")
            std_pp = _f_number(r, f"{schedule}_deployed_std_pp")
            table.append({
                "source": "summary",
                "cell": [r.get("model"), r.get("dataset"), schedule, r.get("depth")],
                "n_seeds": int(r.get("n_seeds") or 0),
                "deployed_acc_mean": mean,
                "std_pp": std_pp,
                "ci95": None,
                "ci_available": False,
                "measured": mean is not None,
                "ann_test_acc_mean": _f_number(r, "ann_test_acc_mean"),
                "validity": r.get("deployment_validity") or r.get("validity"),
                "verdict": r.get("verdict"),
                "item_id": r.get("item_id"),
            })
    return sorted(table, key=lambda r: tuple(
        "" if c is None else str(c) for c in r["cell"]))


# ---------------------------------------------------------------------------
# F2 aggregator — head-to-head delta (percentile-norm - default)
# ---------------------------------------------------------------------------
def aggregate_f2(rows: Iterable[dict]) -> List[dict]:
    """Per-cell delta_pp = percentile-norm mean - default mean, from BOTH schemas.

    Study-tagged: pair the per-seed 0.99 vs 1.0 arms per cell (``source:per_seed``).
    Live summary: the ``quantile_head_to_head`` row carries both arms' means per
    schedule (``{schedule}_deployed_mean_q099``/``_q10``) -> one cell per schedule
    with the delta recomputed from the means (``source:summary``, CI unavailable).
    """
    rows = list(rows)
    table = _aggregate_f2_per_seed(rows)
    table.extend(_aggregate_f2_summary(rows))
    return table


def _aggregate_f2_per_seed(rows: Sequence[dict]) -> List[dict]:
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
            "source": "per_seed",
            "cell": list(key),
            "default_mean": d_mean,
            "default_ci95": d_ci,
            "n_default": len(default[key]),
            "percentile_mean": p_mean,
            "percentile_ci95": p_ci,
            "n_percentile": len(percentile[key]),
            "delta_pp": (p_mean - d_mean) * 100.0,
            "ci_available": True,
        })
    return table


def _aggregate_f2_summary(rows: Sequence[dict]) -> List[dict]:
    table = []
    for r in rows:
        if _summary_study(r) != "F2" or r.get("kind") != "quantile_head_to_head":
            continue
        for schedule in _F_SCHEDULES:
            d_mean = _f_number(r, f"{schedule}_deployed_mean_q099")
            p_mean = _f_number(r, f"{schedule}_deployed_mean_q10")
            if d_mean is None or p_mean is None:
                continue
            table.append({
                "source": "summary",
                "cell": [r.get("model"), r.get("dataset"), schedule, r.get("depth")],
                "default_mean": d_mean,
                "default_ci95": None,
                "n_default": int(r.get("n_seeds") or 0),
                "percentile_mean": p_mean,
                "percentile_ci95": None,
                "n_percentile": int(r.get("n_seeds") or 0),
                "delta_pp": (p_mean - d_mean) * 100.0,
                "ci_available": False,
                "verdict": r.get("verdict"),
                "item_id": r.get("item_id"),
            })
    return sorted(table, key=lambda r: tuple(
        "" if c is None else str(c) for c in r["cell"]))


# ---------------------------------------------------------------------------
# F3 aggregator — dual-regime delta (pretrained - from_scratch)
# ---------------------------------------------------------------------------
F3_CELL_KEYS = ("model", "dataset")


def aggregate_f3(rows: Iterable[dict]) -> List[dict]:
    """Dual-regime delta_pp = pretrained mean - from_scratch mean, BOTH schemas.

    Study-tagged: pair the per-seed from_scratch vs pretrained arms per
    (model,dataset) (``source:per_seed``).
    Live summary: the ``mode`` row carries the from_scratch arm's REAL per-seed
    lists per schedule; the pretrained arm crashed (a ``dual_regime_missing_arms``
    row, ``n_seeds=0``). One cell per schedule surfaces the from_scratch mean and
    emits the pretrained arm + delta as NOT-MEASURED (``delta_measured:False``)
    rather than dropping the cell.
    """
    rows = list(rows)
    table = _aggregate_f3_per_seed(rows)
    table.extend(_aggregate_f3_summary(rows))
    return table


def _aggregate_f3_per_seed(rows: Sequence[dict]) -> List[dict]:
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
            "source": "per_seed",
            "cell": list(key),
            "from_scratch_mean": s_mean,
            "from_scratch_ci95": s_ci,
            "n_from_scratch": len(scratch[key]),
            "pretrained_mean": p_mean,
            "pretrained_ci95": p_ci,
            "n_pretrained": len(pretrained[key]),
            "delta_pp": (p_mean - s_mean) * 100.0,
            "delta_measured": True,
        })
    return table


def _seed_list_mean(row: dict, key: str) -> Tuple[Optional[float], int]:
    vals = row.get(key)
    if not isinstance(vals, list) or not vals:
        return None, 0
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    return (fmean(nums) if nums else None), len(nums)


def _aggregate_f3_summary(rows: Sequence[dict]) -> List[dict]:
    table = []
    for r in rows:
        if _summary_study(r) != "F3" or r.get("kind") != "mode":
            continue
        if bool(r.get("preload_weights")):
            continue  # this branch surfaces the from_scratch arm
        for schedule in _F_SCHEDULES:
            s_mean, n = _seed_list_mean(
                r, f"{schedule}_deployed_full_test_scm_per_seed")
            if s_mean is None:
                s_mean = _f_number(r, f"{schedule}_deployed_mean")
                n = int(r.get("n_seeds") or 0)
            if s_mean is None:
                continue
            table.append({
                "source": "summary",
                "cell": [r.get("model"), r.get("dataset"), schedule],
                "from_scratch_mean": s_mean,
                "from_scratch_ci95": None,
                "n_from_scratch": n,
                "pretrained_mean": None,
                "pretrained_ci95": None,
                "n_pretrained": 0,
                "delta_pp": None,
                "delta_measured": False,
                "ann_test_acc_mean": _f_number(r, "ann_test_acc_mean"),
                "verdict": r.get("verdict"),
                "item_id": r.get("item_id"),
            })
    return sorted(table, key=lambda r: tuple(
        "" if c is None else str(c) for c in r["cell"]))


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


def _fmt_acc(v) -> str:
    return "NOT-MEASURED" if v is None else f"{v:.4f}"


def _fmt_delta(v) -> str:
    return "NOT-MEASURED" if v is None else f"{v:+.2f}"


def _f1_spread(r: dict) -> str:
    """Honest dispersion column: a real CI for per-seed cells, the carried
    std_pp marked CI-unavailable for summary cells, ``n/a`` when neither."""
    if r.get("ci_available") and r.get("ci95") is not None:
        return f"+/- {r['ci95']:.4f}"
    if r.get("std_pp") is not None:
        return f"std {r['std_pp']:.2f}pp (CI-unavailable)"
    return "n/a (CI-unavailable)"


def _f1_md(table: List[dict]) -> str:
    lines = ["## F1 — confidence intervals (per cell, mean +/- 95% CI)", "",
             "| cell (model/dataset/schedule/depth) | n | deployed mean | spread | source |",
             "|---|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {r['n_seeds']} | "
            f"{_fmt_acc(r['deployed_acc_mean'])} | {_f1_spread(r)} | "
            f"{r.get('source', 'per_seed')} |")
    return "\n".join(lines)


def _f2_md(table: List[dict]) -> str:
    lines = ["## F2 — baseline head-to-head (percentile-norm vs default activation-scale)",
             "",
             "| cell | default mean | percentile mean | delta (pp) | source |",
             "|---|---|---|---|---|"]
    for r in table:
        ci = "" if r.get("ci_available") else " (CI-unavailable)"
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {_fmt_acc(r['default_mean'])} | "
            f"{_fmt_acc(r['percentile_mean'])} | {_fmt_delta(r['delta_pp'])}{ci} | "
            f"{r.get('source', 'per_seed')} |")
    return "\n".join(lines)


def _f3_md(table: List[dict]) -> str:
    lines = ["## F3 — dual-regime (from_scratch vs pretrained)", "",
             "| cell | from_scratch mean | pretrained mean | delta (pp) | source |",
             "|---|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {_fmt_cell(r['cell'])} | {_fmt_acc(r['from_scratch_mean'])} | "
            f"{_fmt_acc(r['pretrained_mean'])} | {_fmt_delta(r['delta_pp'])} | "
            f"{r.get('source', 'per_seed')} |")
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
# F3 dual-regime matrix — from_scratch vs pretrained needs a vehicle with a REAL
# pretrained source. The native default_matrix() cells (deep_cnn / lenet5) are
# from-scratch-only: they have no get_pretrained_factory(), so a pretrained arm for
# them is ill-posed (there is no pretrained deep_cnn). torch_squeezenet11 carries
# a torchvision ImageNet factory and is small enough to be a VALID dual-regime cell
# on CIFAR10 (3x32x32) — the cleanest from_scratch-vs-pretrained contrast: both arms
# train the SAME architecture, the only difference is the ImageNet warm-start.
# ---------------------------------------------------------------------------
def f3_dual_regime_matrix() -> List[MatrixCell]:
    """Cells with a real pretrained source for the F3 from_scratch-vs-pretrained delta."""
    return [
        MatrixCell(
            model="torch_squeezenet11", dataset="CIFAR10_DataProvider",
            template="templates/cifar_squeezenet11_dualregime.json",
            schedules=(),
            pretrained_source=PRETRAINED_WEIGHT_SOURCE,
            tags=(("ws", "F"), ("trainable", True), ("regime", "dual"))),
    ]


# ---------------------------------------------------------------------------
# B2 — CIFAR breadth: extend the dataset-margin law to RGB (3x32x32).
#
# The research round flagged CIFAR10 as the next dataset-margin test. These
# cells reuse the MNIST deep_cnn / lenet5 templates and swap the dataset to a
# CIFAR provider through the grid's ``data_provider_name`` axis (exactly how
# the FashionMNIST cell reuses the MNIST template), so no new template file is
# needed. deep_cnn (the VALID trainable-deep vehicle) and lenet5 are both VALID
# on CIFAR at 3x32x32 (param/MAC on-chip fractions clear the majority — verified
# by ``classify_validity`` in the breadth tests). CIFAR100 widens only the head.
# ---------------------------------------------------------------------------
CIFAR_PROVIDERS = ("CIFAR10_DataProvider", "CIFAR100_DataProvider")


def cifar_breadth_matrix() -> List[MatrixCell]:
    """deep_cnn (cascaded+synchronized, d=4/6/8) and lenet5 (synchronized) on
    CIFAR10 and CIFAR100 — the RGB extension of the dataset-margin law."""
    cells: List[MatrixCell] = []
    for dataset in CIFAR_PROVIDERS:
        cells.append(MatrixCell(
            model="deep_cnn", dataset=dataset,
            template="templates/mnist_deep_cnn_d8_cascaded.json",
            schedules=("cascaded", "synchronized"), depths=(4, 6, 8),
            tags=(("ws", "F"), ("trainable", True), ("breadth", "cifar_rgb"))))
        cells.append(MatrixCell(
            model="lenet5", dataset=dataset,
            template="templates/mnist_lenet5_synchronized.json",
            schedules=("synchronized",),
            tags=(("ws", "F"), ("breadth", "cifar_rgb"))))
    return cells


def gen_cifar_breadth_batches(cells: Iterable[MatrixCell], *,
                              seeds: Sequence[int] = (0, 1, 2),
                              priority: int = 38) -> List[dict]:
    """Multi-seed CIFAR breadth cells (study B2) for the dataset-margin estimate."""
    if len(seeds) < 2:
        raise ValueError("CIFAR breadth needs >= 2 seeds for a dataset margin")
    out = []
    for cell in cells:
        grid = _grid(cell, seeds)
        out.append(_batch("B2", "cifar", cell, grid, priority=priority,
                          extra_tags={"kind": "cifar_breadth"}))
    return out


def emit_cifar_breadth_backlog(out_path: str, *,
                               seeds: Sequence[int] = (0, 1, 2)) -> int:
    """Write the CIFAR-breadth batches to ``out_path`` as a backlog JSON list.

    Emits to a FILE only — the orchestrator (not this generator) appends to the
    live backlog. Returns the number of batches written.
    """
    batches = gen_cifar_breadth_batches(cifar_breadth_matrix(), seeds=seeds)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(batches, fh, indent=2)
    return len(batches)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_generate(args) -> int:
    study = args.study.lower()
    if study == "cifar":
        batches = gen_cifar_breadth_batches(cifar_breadth_matrix())
    else:
        cells = default_matrix()
        if study == "f1":
            batches = gen_f1_batches(cells)
        elif study == "f2":
            batches = gen_f2_batches(cells)
        elif study == "f3":
            batches = gen_f3_batches(f3_dual_regime_matrix())
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
    g.add_argument("--study", default="all",
                   choices=["f1", "f2", "f3", "all", "cifar"])
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
