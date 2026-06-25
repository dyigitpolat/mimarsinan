"""E5 — the Pareto DECISION layer: cascaded-vs-synchronized verdict + recipe selector.

This is the program's "automatic genericity" evidence layer. It CONSUMES the campaign's
per-schedule accuracy rows (``cascaded_deployed_mean`` / ``synchronized_deployed_mean`` /
``ann_test_acc_mean`` / ``cascaded_to_sync_gap_pp`` keyed by ``model/dataset/depth/S``)
and a DEFENSIBLE cost PROXY, and emits:

* :func:`pareto_front` — the non-dominated (cost↓ × accuracy↑) subset of decision points;
* :func:`cascaded_vs_synchronized` — the per-dataset deep_cnn verdict (front schedule,
  accuracy gap, cost-band assumption, RETIRE-or-REGIME recommendation);
* :func:`propose_recipe` — the budget-driven recipe selector off the front.

It mutates no simulation state and runs no model. It is a read-only decision layer over
already-extracted campaign data, so it is byte-identical-by-construction with respect to
every deployment path.

THE INTEGRITY CRUX — cost is a MODEL-ESTIMATE WITH A BAND, not measured per-sample energy
-----------------------------------------------------------------------------------------
The ledger carries ``cores_count`` but NOT measured per-sample energy/latency per schedule.
We therefore DO NOT fabricate a cost. Instead:

* **Latency** is derived from the DOCUMENTED genuine-TTFS execution model
  (see ``ttfs_cycle_synchronized_execution`` / ``ttfs_cascade_latch_erosion``):
  synchronized runs latency groups sequentially → ``sim_time = S × latency_groups``;
  cascaded is pipelined → ``sim_time = S + latency_groups``. The only assumption is the
  ``latency_groups ≈ depth`` mapping; we carry a band on it (``±1`` group for the
  input/output framing). This axis is a model-estimate but a TIGHT one (the schedule
  formula is exact given the group count).
* **Energy** is a proxy ``∝ cores × active_steps`` ONLY when ``cores_count`` is present;
  it is reported as a (lo, mid, hi) band and LABELED a model-estimate. The ledger has no
  measured per-sample spike energy per schedule, so absolute per-sample energy is
  UNINSTRUMENTED — :attr:`CostProxyBand.energy_uninstrumented` flags that gap rather than
  inventing a number.

The dominance convention mirrors ``cost_extraction._dominates`` (no-worse on every axis,
strictly better on at least one); :class:`cost_extraction.CostScatter` is the multi-axis
Pareto over measured ``CostRecord``s — this module is the decision-layer altitude above it,
operating on scalar (cost, accuracy) points so the verdict is human-auditable.
"""

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

# A row-level run-id → run-directory resolver. The ledger row carries the
# run-id(s) (``run_id`` / ``cascaded_run_ids`` / ``synchronized_run_ids``) but
# NOT the absolute run directory (it depends on the campaign's
# ``generated_files_path`` at run time), so the caller supplies the mapping.
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

# Verdict recommendation tags.
RETIRE_CASCADED = "RETIRE_CASCADED"
REGIME_DEPENDENT = "REGIME_DEPENDENT"
NO_GAP = "NO_GAP"

_VALID_SCHEDULES = ("cascaded", "synchronized")
_VALID_BUDGETS = ("accuracy", "latency")


# --------------------------------------------------------------------------- #
# pareto_front — scalar (cost, accuracy) dominance, mirroring _dominates.
# --------------------------------------------------------------------------- #

ParetoPoint = Dict[str, object]


def _point_dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Whether ``a`` Pareto-dominates ``b`` on (cost↓, accuracy↑).

    Mirrors ``cost_extraction._dominates``: ``a`` dominates ``b`` iff it is no worse on
    every axis (lower cost, higher accuracy) and strictly better on at least one.
    """
    a_obj = (-float(a["cost"]), float(a["accuracy"]))
    b_obj = (-float(b["cost"]), float(b["accuracy"]))
    no_worse = all(x >= y for x, y in zip(a_obj, b_obj))
    strictly_better = any(x > y for x, y in zip(a_obj, b_obj))
    return no_worse and strictly_better


def pareto_front(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """The non-dominated (cost↓ × accuracy↑) subset of ``points``.

    Each point is a mapping with at least ``cost`` (lower is better) and ``accuracy``
    (higher is better); other keys (e.g. ``label``) ride along untouched. A point is on
    the front iff no OTHER point dominates it (ties keep both — no strict improvement).
    """
    pool = list(points)
    return [
        candidate
        for candidate in pool
        if not any(
            other is not candidate and _point_dominates(other, candidate)
            for other in pool
        )
    ]


# --------------------------------------------------------------------------- #
# schedule_cost_band — the documented latency proxy + energy proxy, with a band.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CostProxyBand:
    """A schedule's cost as a (lo, nominal, hi) BAND of model-estimates, never measured.

    Latency is the documented genuine-TTFS sim-time (exact given the latency-group count,
    banded ±1 group for input/output framing). Energy is the ``cores × active_steps``
    proxy, present only when ``cores`` is known; otherwise :attr:`energy_uninstrumented`
    flags the gap and the energy fields are ``None``.
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
    return int(s_global) + int(groups)  # cascaded pipelined


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
    # Re-order: cascaded and sync are both monotone increasing in group count, so the
    # group band maps to an ordered latency band, but guard against the depth==1 floor.
    lat_lo, lat_hi = min(lat_lo, lat_hi), max(lat_lo, lat_hi)

    if cores is None:
        return CostProxyBand(
            schedule=schedule, s_global=int(s_global), depth=int(depth), cores=None,
            lo_latency_steps=lat_lo, nominal_latency_steps=lat_nom, hi_latency_steps=lat_hi,
            lo_energy_proxy=None, nominal_energy_proxy=None, hi_energy_proxy=None,
        )

    # Energy proxy ~ cores x active_steps (the chip is soma-dominated; every active step
    # charges the resident cores). This is the SAME active-step count as the latency
    # proxy, so the cascaded/sync energy ratio tracks the latency ratio -- a model
    # estimate, not measured spike energy.
    e_lo = float(cores) * lat_lo
    e_nom = float(cores) * lat_nom
    e_hi = float(cores) * lat_hi
    return CostProxyBand(
        schedule=schedule, s_global=int(s_global), depth=int(depth), cores=int(cores),
        lo_latency_steps=lat_lo, nominal_latency_steps=lat_nom, hi_latency_steps=lat_hi,
        lo_energy_proxy=e_lo, nominal_energy_proxy=e_nom, hi_energy_proxy=e_hi,
    )


# --------------------------------------------------------------------------- #
# load_measured_cost — PREFER a measured cost_record over the proxy when one
# resolves for a row's run_ids (the cost-emit unlock); else None (proxy stays).
# --------------------------------------------------------------------------- #

# The row run-id fields, in schedule order, a measured record can hang off.
_RUN_ID_FIELDS_BY_SCHEDULE = {
    "cascaded": ("cascaded_run_ids",),
    "synchronized": ("synchronized_run_ids",),
}
# Schedule-agnostic run-id fields tried for ANY schedule (a single-schedule row).
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
    """The MEASURED :class:`CostRecord` for a row's ``schedule``, or ``None``.

    Resolves the row's run-id(s) for ``schedule`` to run directories via
    ``run_dir_resolver`` and loads the first resolvable ``cost_record.json``
    (the measured cost the deployment now emits, cost-emit). Returns ``None``
    when no resolver is supplied or no record resolves — the caller then falls
    back to the documented cost PROXY, keeping the proxy path byte-identical.

    THE RESOLUTION GAP (documented, not guessed): the ledger row carries the
    run-id(s) but NOT the absolute run directory (it depends on the campaign's
    ``generated_files_path`` at run time). The mapping is therefore the caller's
    responsibility via ``run_dir_resolver``; without one this returns ``None``
    and E5 stays on the proxy.
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


# --------------------------------------------------------------------------- #
# cascaded_vs_synchronized — the per-dataset verdict.
# --------------------------------------------------------------------------- #

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
    accuracy_gap_pp: float            # synchronized - cascaded, in percentage points
    front_schedule: str               # the accuracy-front schedule
    front_points: List[ParetoPoint]   # (latency-cost, accuracy) points on the front
    cost_band: CostProxyBand          # the synchronized cost band (the dearer schedule)
    cascaded_cost_band: CostProxyBand
    recommendation: str               # RETIRE_CASCADED / REGIME_DEPENDENT / NO_GAP
    conditional_on_cost_band: bool
    # The (cost, accuracy) decision points the front was computed over. The
    # ``cost`` is the MEASURED ``latency_steps`` when a cost_record resolved for
    # the schedule's run-id(s), else the documented proxy ``nominal_latency_steps``.
    decision_points: List[ParetoPoint] = field(default_factory=list)
    # Whether the decision points used a MEASURED cost record (vs the proxy).
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


_GAP_NOISE_PP = 0.5  # below this the schedules are tied (small-N read noise floor)


def _cell_rank(row: dict) -> tuple:
    """A reproducible preference key over duplicate-cell measurements (higher wins).

    The ledger is append-only with re-measurements of the same cell. We prefer the most
    DEFENSIBLE row: (1) energy-instrumented (``cores_count`` present, so the energy axis
    is real not UNINSTRUMENTED), then (2) the deepest cell (depth stresses the cascade),
    then (3) the latest timestamp (the reconciled measurement). This lands the verdict on
    the most-instrumented recent cell deterministically.
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

    # The (latency-cost, accuracy) decision points -- latency is the deterministic
    # discriminating cost axis (energy tracks it). The cost is the MEASURED
    # latency_steps when a cost_record resolves for the schedule's run-id(s)
    # (cost-emit), else the documented proxy band (byte-identical fallback).
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
        front_schedule = "synchronized"  # tie -> name the safe (accuracy) default
        recommendation = NO_GAP
    elif gap_pp > 0:
        front_schedule = "synchronized"
        # Synchronized wins accuracy. It RETIRES cascaded only if it ALSO dominates on
        # cost -- i.e. cascaded is NOT on the front. Cascaded's strictly-lower pipelined
        # latency keeps it on the (cost, accuracy) front, so the honest verdict is a
        # REGIME (cascaded wins the hard-latency budget), CONDITIONAL on the cost band.
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
    """The cascaded-vs-synchronized verdict per deep_cnn dataset.

    Consumes campaign science rows (any ``kind``) carrying both schedule accuracies;
    for each dataset it takes the deepest measured cell (depth stresses the cascade) and
    emits a :class:`ScheduleVerdict`. The verdict is CONDITIONAL on the cost-proxy band:
    synchronized is RETIRE-worthy only if it dominates on accuracy AND cost; cascaded's
    lower pipelined latency otherwise keeps the choice REGIME-dependent.

    ``run_dir_resolver`` (cost-emit) optionally maps a row's run-id(s) to run
    directories so the decision cost prefers the MEASURED ``latency_steps`` of an
    emitted ``cost_record.json`` over the proxy. When ``None`` (the default) the
    decision uses the documented cost proxy — byte-identical to the pre-cost-emit
    path.
    """
    deep_cnn_rows = [r for r in rows if r.get("model") == "deep_cnn"]
    best = _best_cell_per_dataset(deep_cnn_rows)
    per_dataset = {
        ds: _verdict_for_cell(row, run_dir_resolver=run_dir_resolver)
        for ds, row in best.items()
    }
    return CascadeVsSyncVerdict(per_dataset=per_dataset)


# --------------------------------------------------------------------------- #
# propose_recipe — E5c budget-driven selector off the front.
# --------------------------------------------------------------------------- #

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
    """The E5c recipe selector: pick (firing, schedule, S, placement) for ``budget``.

    ``accuracy`` budget → synchronized (the accuracy-front schedule). ``latency`` budget
    → cascaded IF it is on the (latency-cost, accuracy) front, else synchronized (a
    lower-accuracy AND not-cheaper cascaded is never chosen). The pick is CONDITIONAL on
    the cost-proxy band (see :data:`COST_BAND_DISCLAIMER`).
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

    # hard-latency budget
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


# --------------------------------------------------------------------------- #
# load_deep_cnn_rows — read the campaign ledger (own reader; json fallback).
# --------------------------------------------------------------------------- #

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
