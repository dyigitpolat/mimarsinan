"""Pareto decision layer over campaign rows: cascaded-vs-synchronized verdict + recipe selector."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from mimarsinan.chip_simulation.cost_extraction import (
    COST_RECORD_FILENAME,
    CostRecord,
    load_cost_record,
)

logger = logging.getLogger("mimarsinan.chip_simulation")

RunDirResolver = Callable[[str], Optional[str]]


__all__ = [
    "CostProxyBand",
    "ParetoPoint",
    "ScheduleVerdict",
    "CascadeVsSyncVerdict",
    "RecipeProposal",
    "RunDirResolver",
    "COST_BAND_DISCLAIMER",
    "pareto_front",
    "schedule_cost_band",
    "load_measured_cost",
    "cascaded_vs_synchronized",
    "propose_recipe",
    "load_deep_cnn_rows",
]


COST_BAND_DISCLAIMER = (
    "Cost is a MODEL-ESTIMATE WITH A BAND derived from the documented genuine-TTFS "
    "execution model (synchronized sim_time = S x latency_groups; cascaded pipelined = "
    "S + latency_groups) and an energy proxy ~ cores x active_steps; it is NOT measured "
    "per-sample energy. Per-sample spike energy per schedule is UNINSTRUMENTED in the "
    "ledger -- a stated remaining instrumentation gap."
)

RETIRE_CASCADED = "RETIRE_CASCADED"
REGIME_DEPENDENT = "REGIME_DEPENDENT"
NO_GAP = "NO_GAP"

_VALID_SCHEDULES = ("cascaded", "synchronized")
_VALID_BUDGETS = ("accuracy", "latency")


ParetoPoint = Dict[str, object]


def _point_dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Whether ``a`` Pareto-dominates ``b`` on (cost↓, accuracy↑; strictly better on one)."""
    a_obj = (-float(a["cost"]), float(a["accuracy"]))
    b_obj = (-float(b["cost"]), float(b["accuracy"]))
    no_worse = all(x >= y for x, y in zip(a_obj, b_obj))
    strictly_better = any(x > y for x, y in zip(a_obj, b_obj))
    return no_worse and strictly_better


def pareto_front(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """The non-dominated (cost↓ × accuracy↑) subset of ``points`` (ties keep both)."""
    pool = list(points)
    return [
        candidate
        for candidate in pool
        if not any(
            other is not candidate and _point_dominates(other, candidate)
            for other in pool
        )
    ]


@dataclass(frozen=True)
class CostProxyBand:
    """A schedule's cost as a (lo, nominal, hi) BAND of model-estimates, never measured.

    Energy is the ``cores × active_steps`` proxy, present only when ``cores`` is known;
    otherwise :attr:`energy_uninstrumented` flags the gap and the energy fields are ``None``.
    """

    schedule: str
    s_global: int
    depth: int
    cores: Optional[int]
    lo_latency_steps: int
    nominal_latency_steps: int
    hi_latency_steps: int
    lo_energy_proxy: Optional[float]
    nominal_energy_proxy: Optional[float]
    hi_energy_proxy: Optional[float]
    is_model_estimate: bool = True
    disclaimer: str = COST_BAND_DISCLAIMER

    @property
    def energy_uninstrumented(self) -> bool:
        """True when no measured/derivable energy proxy exists (``cores`` absent)."""
        return self.nominal_energy_proxy is None


def _latency_groups_band(depth: int) -> tuple:
    """The (lo, nominal, hi) latency-group count: ``depth`` ± 1 for I/O framing."""
    nominal = max(int(depth), 1)
    lo = max(nominal - 1, 1)
    hi = nominal + 1
    return lo, nominal, hi


def _sim_time(schedule: str, s_global: int, groups: int) -> int:
    """The documented sim-time for ``schedule``: sync = S·groups; cascaded = S+groups."""
    if schedule == "synchronized":
        return int(s_global) * int(groups)
    return int(s_global) + int(groups)


def schedule_cost_band(
    schedule: str,
    *,
    s_global: int,
    depth: int,
    cores: Optional[int] = None,
) -> CostProxyBand:
    """The DEFENSIBLE cost-proxy band for one schedule at ``(S, depth, cores)``.

    Latency from the documented execution model; energy ~ ``cores × active_steps`` (the
    same ``Σ_d steps`` the latency proxy counts) when ``cores`` is known, else flagged
    UNINSTRUMENTED. Both are model-estimates carried as a (lo, mid, hi) band.
    """
    if schedule not in _VALID_SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {_VALID_SCHEDULES}")

    g_lo, g_nom, g_hi = _latency_groups_band(depth)
    lat_lo = _sim_time(schedule, s_global, g_lo)
    lat_nom = _sim_time(schedule, s_global, g_nom)
    lat_hi = _sim_time(schedule, s_global, g_hi)
    lat_lo, lat_hi = min(lat_lo, lat_hi), max(lat_lo, lat_hi)

    if cores is None:
        return CostProxyBand(
            schedule=schedule, s_global=int(s_global), depth=int(depth), cores=None,
            lo_latency_steps=lat_lo, nominal_latency_steps=lat_nom, hi_latency_steps=lat_hi,
            lo_energy_proxy=None, nominal_energy_proxy=None, hi_energy_proxy=None,
        )

    e_lo = float(cores) * lat_lo
    e_nom = float(cores) * lat_nom
    e_hi = float(cores) * lat_hi
    return CostProxyBand(
        schedule=schedule, s_global=int(s_global), depth=int(depth), cores=int(cores),
        lo_latency_steps=lat_lo, nominal_latency_steps=lat_nom, hi_latency_steps=lat_hi,
        lo_energy_proxy=e_lo, nominal_energy_proxy=e_nom, hi_energy_proxy=e_hi,
    )


_RUN_ID_FIELDS_BY_SCHEDULE = {
    "cascaded": ("cascaded_run_ids",),
    "synchronized": ("synchronized_run_ids",),
}
_RUN_ID_FIELDS_ANY = ("run_id", "run_ids")


def _run_ids_for_schedule(row: dict, schedule: str) -> List[str]:
    """The row's run-id(s) for ``schedule`` (schedule-specific then generic)."""
    fields = _RUN_ID_FIELDS_BY_SCHEDULE.get(schedule, ()) + _RUN_ID_FIELDS_ANY
    ids: List[str] = []
    for field_name in fields:
        value = row.get(field_name)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        ids.extend(str(c) for c in candidates if c)
    return ids


def load_measured_cost(
    row: dict,
    *,
    schedule: str,
    run_dir_resolver: Optional[RunDirResolver] = None,
) -> Optional[CostRecord]:
    """The MEASURED :class:`CostRecord` for a row's ``schedule``, or ``None`` (proxy fallback).

    Resolves the row's run-id(s) via ``run_dir_resolver`` and loads the first resolvable
    ``cost_record.json``; without a resolver or record this returns ``None``.
    """
    if run_dir_resolver is None:
        return None
    for run_id in _run_ids_for_schedule(row, schedule):
        try:
            run_dir = run_dir_resolver(run_id)
        except Exception:
            logger.exception("run_dir_resolver raised for run_id %r", run_id)
            continue
        if not run_dir:
            continue
        path = os.path.join(run_dir, COST_RECORD_FILENAME)
        if not os.path.exists(path):
            continue
        try:
            return load_cost_record(path)
        except Exception:
            logger.exception("failed to load cost record at %s", path)
            continue
    return None


@dataclass
class ScheduleVerdict:
    """The cascaded-vs-synchronized verdict for ONE deep_cnn dataset cell."""

    dataset: str
    depth: int
    s_global: int
    cores: Optional[int]
    cascaded_accuracy: float
    synchronized_accuracy: float
    ann_accuracy: Optional[float]
    accuracy_gap_pp: float
    front_schedule: str
    front_points: List[ParetoPoint]
    cost_band: CostProxyBand
    cascaded_cost_band: CostProxyBand
    recommendation: str
    conditional_on_cost_band: bool
    decision_points: List[ParetoPoint] = field(default_factory=list)
    cost_measured: bool = False
    cost_band_assumption: str = COST_BAND_DISCLAIMER

    @property
    def synchronized_dominates(self) -> bool:
        """True iff synchronized is on the front AND cascaded is NOT (full domination)."""
        labels = {p["label"] for p in self.front_points}
        return "synchronized" in labels and "cascaded" not in labels


@dataclass
class CascadeVsSyncVerdict:
    """The program-level verdict: per-dataset :class:`ScheduleVerdict`s + a roll-up."""

    per_dataset: Dict[str, ScheduleVerdict] = field(default_factory=dict)
    cost_band_assumption: str = COST_BAND_DISCLAIMER

    @property
    def any_full_retire(self) -> bool:
        """True iff synchronized DOMINATES (acc AND cost) on every measured dataset."""
        if not self.per_dataset:
            return False
        return all(v.recommendation == RETIRE_CASCADED for v in self.per_dataset.values())


_GAP_NOISE_PP = 0.5


def _cell_rank(row: dict) -> tuple:
    """A reproducible preference key over duplicate-cell measurements (higher wins).

    Prefers the most defensible row: energy-instrumented, then deepest, then latest.
    """
    has_cores = 1 if row.get("cores_count") is not None else 0
    depth = int(row.get("depth") or 0)
    ts = float(row.get("ts") or 0.0)
    return (has_cores, depth, ts)


def _best_cell_per_dataset(rows: Sequence[dict]) -> Dict[str, dict]:
    """The most-defensible measured deep_cnn cell per dataset (see :func:`_cell_rank`)."""
    best: Dict[str, dict] = {}
    for r in rows:
        ds = _norm_dataset(r.get("dataset"))
        if ds is None:
            continue
        if r.get("cascaded_deployed_mean") is None or r.get("synchronized_deployed_mean") is None:
            continue
        if r.get("depth") is None:
            continue
        cur = best.get(ds)
        if cur is None or _cell_rank(r) > _cell_rank(cur):
            best[ds] = r
    return best


def _norm_dataset(name) -> Optional[str]:
    """Canonicalize dataset labels (``MNIST``/``mnist``/``fashion_mnist`` -> a tag)."""
    if name is None:
        return None
    low = str(name).strip().lower()
    aliases = {
        "fashion_mnist": "fmnist",
        "fashionmnist": "fmnist",
    }
    return aliases.get(low, low)


def _schedule_cost(
    row: dict,
    schedule: str,
    proxy_band: CostProxyBand,
    run_dir_resolver: Optional[RunDirResolver],
) -> tuple:
    """``(latency_cost, used_measured)`` for one schedule: PREFER the measured
    cost_record's ``latency_steps``, else the documented proxy ``nominal_latency_steps``."""
    measured = load_measured_cost(
        row, schedule=schedule, run_dir_resolver=run_dir_resolver,
    )
    if measured is not None:
        return int(measured.latency_steps), True
    return int(proxy_band.nominal_latency_steps), False


def _verdict_for_cell(
    row: dict, *, run_dir_resolver: Optional[RunDirResolver] = None,
) -> ScheduleVerdict:
    ds = _norm_dataset(row["dataset"])
    depth = int(row["depth"])
    s_global = int(row.get("S") or 4)
    cores = row.get("cores_count")
    casc = float(row["cascaded_deployed_mean"])
    sync = float(row["synchronized_deployed_mean"])
    ann = row.get("ann_test_acc_mean")
    gap_pp = (sync - casc) * 100.0

    casc_band = schedule_cost_band("cascaded", s_global=s_global, depth=depth, cores=cores)
    sync_band = schedule_cost_band("synchronized", s_global=s_global, depth=depth, cores=cores)

    casc_cost, casc_measured = _schedule_cost(row, "cascaded", casc_band, run_dir_resolver)
    sync_cost, sync_measured = _schedule_cost(row, "synchronized", sync_band, run_dir_resolver)
    cost_measured = casc_measured or sync_measured
    points = [
        {"label": "cascaded", "cost": casc_cost, "accuracy": casc},
        {"label": "synchronized", "cost": sync_cost, "accuracy": sync},
    ]
    front = pareto_front(points)
    front_labels = {p["label"] for p in front}

    if abs(gap_pp) < _GAP_NOISE_PP:
        front_schedule = "synchronized"
        recommendation = NO_GAP
    elif gap_pp > 0:
        front_schedule = "synchronized"
        recommendation = RETIRE_CASCADED if "cascaded" not in front_labels else REGIME_DEPENDENT
    else:
        front_schedule = "cascaded"
        recommendation = REGIME_DEPENDENT

    return ScheduleVerdict(
        dataset=ds, depth=depth, s_global=s_global, cores=cores,
        cascaded_accuracy=casc, synchronized_accuracy=sync,
        ann_accuracy=float(ann) if ann is not None else None,
        accuracy_gap_pp=round(gap_pp, 2),
        front_schedule=front_schedule, front_points=front,
        cost_band=sync_band, cascaded_cost_band=casc_band,
        recommendation=recommendation,
        conditional_on_cost_band=True,
        decision_points=points,
        cost_measured=cost_measured,
    )


def cascaded_vs_synchronized(
    rows: Sequence[dict],
    *,
    run_dir_resolver: Optional[RunDirResolver] = None,
) -> CascadeVsSyncVerdict:
    """The cascaded-vs-synchronized verdict per deep_cnn dataset, over the deepest measured cell.

    CONDITIONAL on the cost-proxy band: synchronized is RETIRE-worthy only if it dominates
    on accuracy AND cost. ``run_dir_resolver`` optionally prefers a MEASURED cost over the proxy.
    """
    deep_cnn_rows = [r for r in rows if r.get("model") == "deep_cnn"]
    best = _best_cell_per_dataset(deep_cnn_rows)
    per_dataset = {
        ds: _verdict_for_cell(row, run_dir_resolver=run_dir_resolver)
        for ds, row in best.items()
    }
    return CascadeVsSyncVerdict(per_dataset=per_dataset)


@dataclass(frozen=True)
class RecipeProposal:
    """A deployment recipe picked from the front for a stated budget."""

    budget: str
    dataset: str
    firing: str
    schedule: str
    s_global: int
    placement: str
    rationale: str


def propose_recipe(
    budget: str,
    *,
    rows: Sequence[dict],
    dataset: str,
) -> RecipeProposal:
    """Pick (firing, schedule, S, placement) for ``budget`` off the front.

    ``accuracy`` → synchronized; ``latency`` → cascaded iff it is on the front, else
    synchronized. CONDITIONAL on the cost-proxy band (see :data:`COST_BAND_DISCLAIMER`).
    """
    if budget not in _VALID_BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; expected one of {_VALID_BUDGETS}")

    verdict = cascaded_vs_synchronized(rows)
    ds = _norm_dataset(dataset)
    cell = verdict.per_dataset.get(ds)
    if cell is None:
        raise ValueError(f"no deep_cnn verdict for dataset {dataset!r}")

    s_global = cell.s_global
    placement = "on_chip_majority"
    firing = "ttfs"

    if budget == "accuracy":
        return RecipeProposal(
            budget=budget, dataset=ds, firing=firing, schedule="synchronized",
            s_global=s_global, placement=placement,
            rationale=(
                f"accuracy-priority: synchronized is the accuracy-front schedule "
                f"(+{cell.accuracy_gap_pp:.2f}pp over cascaded). {COST_BAND_DISCLAIMER}"
            ),
        )

    front_labels = {p["label"] for p in cell.front_points}
    cascaded_on_front = "cascaded" in front_labels
    schedule = "cascaded" if cascaded_on_front else "synchronized"
    rationale = (
        f"hard-latency: cascaded is on the (latency, accuracy) front "
        f"(pipelined sim_time S+groups < S*groups); chosen despite a "
        f"{cell.accuracy_gap_pp:.2f}pp accuracy cost. {COST_BAND_DISCLAIMER}"
        if cascaded_on_front
        else (
            f"hard-latency: cascaded is OFF the front (no latency win at depth "
            f"{cell.depth}) and lower accuracy -> fall back to synchronized. "
            f"{COST_BAND_DISCLAIMER}"
        )
    )
    return RecipeProposal(
        budget=budget, dataset=ds, firing=firing, schedule=schedule,
        s_global=s_global, placement=placement, rationale=rationale,
    )


def load_deep_cnn_rows(ledger_path: str) -> List[dict]:
    """Mine the deep_cnn rows carrying BOTH schedule accuracies from a JSONL ledger."""
    rows: List[dict] = []
    with open(ledger_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("model") != "deep_cnn":
                continue
            if rec.get("cascaded_deployed_mean") is None:
                continue
            if rec.get("synchronized_deployed_mean") is None:
                continue
            rows.append(rec)
    return rows
