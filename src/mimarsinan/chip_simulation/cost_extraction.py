"""Mine the cost record a sim already emits into a cell-keyed accuracy×cost scatter."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.weight_reuse_cost_model import (
    DEFAULT_COEFFICIENT_BAND,
    CoefficientBand,
    phase_cost_band,
)


__all__ = [
    "CostRecord",
    "CostScatter",
    "COST_RECORD_FORMAT_VERSION",
    "COST_RECORD_FILENAME",
    "FT_PASS_WALLS_FILENAME",
    "extract_cost_record",
    "extract_cost_record_from_run",
    "load_cost_record",
    "save_cost_record",
    "weight_reuse_mj",
]


COST_RECORD_FORMAT_VERSION = 3


def weight_reuse_mj(
    *,
    params_reloaded: int,
    activation_bytes_moved: int,
    mj_per_reprogram: float = 0.0,
    mj_per_sync: float = 0.0,
) -> float:
    """The weight-reuse scheduling mJ term: reprogram param reload + sync byte movement.

    ``mj_per_reprogram`` charges per reloaded param, ``mj_per_sync`` per moved activation
    byte; both default 0.0 ⇒ the term is 0.0 (byte-identical). See ``weight_reuse_cost_model``
    for the defensible decomposed coefficients + uncertainty band.
    """
    return (
        float(mj_per_reprogram) * float(params_reloaded)
        + float(mj_per_sync) * float(activation_bytes_moved)
    )

COST_RECORD_FILENAME = "cost_record.json"
FT_PASS_WALLS_FILENAME = "ft_pass_walls.json"


def _neuron_steps_from_sanafe_snapshot(snapshot: Mapping[str, Any]) -> Tuple[int, int]:
    """``(Σ_d neurons_d · S_d, total cores)`` over a SANA-FE snapshot's segments."""
    per_sample = snapshot.get("per_sample") or []
    if not per_sample:
        return 0, 0
    segments = per_sample[0].get("segments") or []
    neuron_steps = 0
    cores = 0
    for seg in segments:
        per_core = seg.get("per_core") or []
        neurons = sum(int(c.get("n_neurons", 0)) for c in per_core)
        timesteps = int(seg.get("timesteps_executed", 0))
        neuron_steps += neurons * timesteps
        cores += len(per_core)
    return neuron_steps, cores


def _latency_steps_from_sanafe_snapshot(snapshot: Mapping[str, Any]) -> int:
    """The deployed latency in temporal steps: ``Σ_d timesteps_executed_d``."""
    per_sample = snapshot.get("per_sample") or []
    if not per_sample:
        return 0
    segments = per_sample[0].get("segments") or []
    return sum(int(seg.get("timesteps_executed", 0)) for seg in segments)


def _global_s_from_sanafe_snapshot(snapshot: Mapping[str, Any]) -> int:
    """The global temporal resolution S (= the per-sample ``T``)."""
    per_sample = snapshot.get("per_sample") or []
    if not per_sample:
        return 0
    return int(per_sample[0].get("T", 0))


def _depth_from_sanafe_snapshot(snapshot: Mapping[str, Any]) -> int:
    """The cascade depth = number of neural segments / latency groups."""
    per_sample = snapshot.get("per_sample") or []
    if not per_sample:
        return 0
    return len(per_sample[0].get("segments") or [])


def _coerce_ft_pass_walls(
    walls: Sequence[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], ...]:
    """Normalize an FT-pass breakdown to a hashable tuple of ``{label, wall_s}`` dicts."""
    return tuple(
        {"label": str(p.get("label", "")), "wall_s": float(p.get("wall_s", 0.0))}
        for p in (walls or ())
    )


@dataclass(frozen=True)
class CostRecord:
    """The cost tuple a sim run emits, keyed to its E6 (firing × sync × backend) cell.

    Axes: ``acc_deploy`` / ``mj_per_sample`` / ``spikes`` / ``latency_steps`` /
    ``cores`` / ``s_global`` / ``depth``. The weight-reuse scheduling fields and
    ``max_ft_pass_wall_s`` all default 0 ⇒ byte-identical to a pre-mode record.
    """

    cell_key: str
    mode: str
    backend: str
    acc_deploy: float
    mj_per_sample: float
    spikes: int
    latency_steps: int
    cores: int
    s_global: int
    depth: int
    energy_proxy_neuron_steps: int = 0
    max_ft_pass_wall_s: float = 0.0
    ft_pass_walls: Tuple[Mapping[str, Any], ...] = ()
    reprogram_passes: int = 0
    reuse_passes: int = 0
    params_reloaded: int = 0
    activation_bytes_moved: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = COST_RECORD_FORMAT_VERSION

    @property
    def cell(self) -> CertificationCell:
        """The E6 cell this record is keyed to (parsed from ``cell_key``)."""
        return CertificationCell.from_key(self.cell_key)

    @property
    def total_passes(self) -> int:
        """Total scheduled passes = reprogram (N) + reuse (M)."""
        return self.reprogram_passes + self.reuse_passes

    @property
    def sync_barrier_count(self) -> int:
        """Activation-gather barriers = ``total_passes - 1`` (0 when no passes)."""
        return max(self.total_passes - 1, 0)

    @property
    def reuse_fraction(self) -> float:
        """Fraction of passes that reuse a resident bank (0.0 when no passes)."""
        if self.total_passes == 0:
            return 0.0
        return self.reuse_passes / self.total_passes

    def reuse_mj(
        self, *, mj_per_reprogram: float = 0.0, mj_per_sync: float = 0.0
    ) -> float:
        """The weight-reuse scheduling mJ term at the given chip coefficients.

        Default 0.0 coefficients ⇒ 0.0 (byte-identical). See :func:`weight_reuse_mj`.
        """
        return weight_reuse_mj(
            params_reloaded=self.params_reloaded,
            activation_bytes_moved=self.activation_bytes_moved,
            mj_per_reprogram=mj_per_reprogram,
            mj_per_sync=mj_per_sync,
        )

    def reuse_mj_band(
        self, coeffs: CoefficientBand = DEFAULT_COEFFICIENT_BAND
    ) -> Tuple[float, float, float]:
        """The GROUNDED weight-reuse mJ as a ``(low, nominal, high)`` RANGE (0.0s when no passes)."""
        band = phase_cost_band(
            reprogram_passes=self.reprogram_passes,
            reuse_passes=self.reuse_passes,
            params_reloaded=self.params_reloaded,
            activation_bytes_moved=self.activation_bytes_moved,
            coefficient_band=coeffs,
        )
        return (band.low_mj, band.nominal_mj, band.high_mj)

    def cost_tuple(self) -> Tuple[float, int, int, int]:
        """The cost axes ``(mj_per_sample, spikes, latency_steps, cores)``."""
        return (self.mj_per_sample, self.spikes, self.latency_steps, self.cores)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provenance"] = dict(self.provenance)
        data["ft_pass_walls"] = [dict(p) for p in self.ft_pass_walls]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostRecord":
        version = int(data.get("format_version", COST_RECORD_FORMAT_VERSION))
        if version != COST_RECORD_FORMAT_VERSION:
            raise ValueError(
                f"cost-record format_version {version} != "
                f"{COST_RECORD_FORMAT_VERSION}; the format changed — re-extract"
            )
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"cost record has unknown fields {sorted(unknown)} — the format may "
                f"have drifted; re-extract the record"
            )
        kwargs = dict(data)
        kwargs["provenance"] = dict(kwargs.get("provenance", {}))
        kwargs["ft_pass_walls"] = _coerce_ft_pass_walls(kwargs.get("ft_pass_walls", ()))
        return cls(**kwargs)


def extract_cost_record(
    *,
    cell: CertificationCell,
    deployed_accuracy: float,
    sanafe_snapshot: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
    max_ft_pass_wall_s: float = 0.0,
    ft_pass_walls: Sequence[Mapping[str, Any]] = (),
    reprogram_passes: int = 0,
    reuse_passes: int = 0,
    params_reloaded: int = 0,
    activation_bytes_moved: int = 0,
    coefficient_band: Optional[CoefficientBand] = None,
) -> CostRecord:
    """Build a :class:`CostRecord` from a SANA-FE snapshot dict — a pure read, runs nothing.

    ``coefficient_band`` is opt-in: when supplied the record carries the grounded
    weight-reuse mJ range under ``provenance['reuse_cost_band_mj']``; ``None`` (default)
    attaches nothing and stays byte-identical to a pre-band record.
    """
    aggregate = sanafe_snapshot.get("aggregate") or {}
    mj_per_sample = float(aggregate.get("total_energy_mj", 0.0))
    sample_count = int(aggregate.get("sample_count", 0) or 0)
    if sample_count > 1:
        mj_per_sample = mj_per_sample / sample_count
    spikes = int(aggregate.get("total_spikes", 0))

    neuron_steps, cores = _neuron_steps_from_sanafe_snapshot(sanafe_snapshot)
    latency_steps = _latency_steps_from_sanafe_snapshot(sanafe_snapshot)
    s_global = _global_s_from_sanafe_snapshot(sanafe_snapshot)
    depth = _depth_from_sanafe_snapshot(sanafe_snapshot)

    mode = cell.cell_key.rsplit("@", 1)[0]

    record = CostRecord(
        cell_key=cell.cell_key,
        mode=mode,
        backend=cell.backend,
        acc_deploy=float(deployed_accuracy),
        mj_per_sample=mj_per_sample,
        spikes=spikes,
        latency_steps=latency_steps,
        cores=cores,
        s_global=s_global,
        depth=depth,
        energy_proxy_neuron_steps=neuron_steps,
        max_ft_pass_wall_s=float(max_ft_pass_wall_s),
        ft_pass_walls=_coerce_ft_pass_walls(ft_pass_walls),
        reprogram_passes=int(reprogram_passes),
        reuse_passes=int(reuse_passes),
        params_reloaded=int(params_reloaded),
        activation_bytes_moved=int(activation_bytes_moved),
        provenance=dict(provenance or {}),
    )
    return _with_grounded_cost_band(record, coefficient_band)


def _with_grounded_cost_band(
    record: CostRecord, coefficient_band: Optional[CoefficientBand]
) -> CostRecord:
    """Attach the grounded weight-reuse mJ band to ``record`` ONLY when configured.

    ``coefficient_band is None`` (the default) returns ``record`` UNCHANGED, byte-identical.
    """
    if coefficient_band is None:
        return record
    low, nominal, high = record.reuse_mj_band(coefficient_band)
    provenance = dict(record.provenance)
    provenance["reuse_cost_band_mj"] = [low, nominal, high]
    return replace(record, provenance=provenance)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _deployed_accuracy_from_run(run_dir: str) -> float:
    """The deployed metric a run persisted to ``__target_metric.json``."""
    target_path = os.path.join(run_dir, "__target_metric.json")
    if not os.path.exists(target_path):
        return 0.0
    return float(_read_json(target_path))


def _sanafe_snapshot_from_run(run_dir: str) -> Optional[Mapping[str, Any]]:
    """The persisted SANA-FE snapshot dict (``None`` when the run carried no SANA-FE step)."""
    steps_path = os.path.join(run_dir, "_GUI_STATE", "steps.json")
    if not os.path.exists(steps_path):
        return None
    steps = (_read_json(steps_path) or {}).get("steps") or {}
    sanafe_step = steps.get("SANA-FE Simulation") or {}
    snapshot = sanafe_step.get("snapshot") or {}
    return snapshot.get("sanafe_simulation")


def _ft_pass_walls_from_run(run_dir: str) -> Tuple[float, Tuple[Mapping[str, Any], ...]]:
    """The AC5 per-fine-tuning-PASS wall bundle a run persisted (``(0.0, ())`` when none)."""
    path = os.path.join(run_dir, FT_PASS_WALLS_FILENAME)
    if not os.path.exists(path):
        return 0.0, ()
    data = _read_json(path) or {}
    return (
        float(data.get("max_ft_pass_wall_s", 0.0)),
        _coerce_ft_pass_walls(data.get("passes", ())),
    )


def _cell_from_run_config(run_dir: str, *, backend: str) -> Optional[CertificationCell]:
    """The (firing × sync) cell declared by a generated run's ``_RUN_CONFIG``, via the ``DeploymentPlan`` SSOT."""
    config_path = os.path.join(run_dir, "_RUN_CONFIG", "config.json")
    if not os.path.exists(config_path):
        return None
    config = _read_json(config_path) or {}
    params = config.get("deployment_parameters") or {}
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    plan = DeploymentPlan.resolve(params)
    return CertificationCell.from_mode_policy(plan.mode_policy(), backend=backend)


def extract_cost_record_from_run(
    run_dir: str, *, backend: str = "sanafe"
) -> Optional[CostRecord]:
    """Mine a generated-run directory's persisted artifacts into a :class:`CostRecord` (``None`` when no SANA-FE cost data)."""
    snapshot = _sanafe_snapshot_from_run(run_dir)
    if snapshot is None:
        return None
    cell = _cell_from_run_config(run_dir, backend=backend)
    if cell is None:
        return None
    deployed_accuracy = _deployed_accuracy_from_run(run_dir)
    max_ft_pass_wall_s, ft_pass_walls = _ft_pass_walls_from_run(run_dir)
    return extract_cost_record(
        cell=cell,
        deployed_accuracy=deployed_accuracy,
        sanafe_snapshot=snapshot,
        provenance={"run_dir": os.path.basename(os.path.normpath(run_dir))},
        max_ft_pass_wall_s=max_ft_pass_wall_s,
        ft_pass_walls=ft_pass_walls,
    )


def save_cost_record(record: CostRecord, run_dir: str) -> str:
    """Write a cost record alongside the run artifacts; return the written path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, COST_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_cost_record(path: str) -> CostRecord:
    """Load a cost record from a JSON file (the format a run wrote)."""
    return CostRecord.from_dict(_read_json(path))


def _dominates(a: CostRecord, b: CostRecord) -> bool:
    """Whether ``a`` Pareto-dominates ``b`` (accuracy↑, every cost axis↓; strictly better on one)."""
    a_obj = (a.acc_deploy, -a.mj_per_sample, -a.spikes, -a.latency_steps, -a.cores)
    b_obj = (b.acc_deploy, -b.mj_per_sample, -b.spikes, -b.latency_steps, -b.cores)
    no_worse = all(x >= y for x, y in zip(a_obj, b_obj))
    strictly_better = any(x > y for x, y in zip(a_obj, b_obj))
    return no_worse and strictly_better


@dataclass
class CostScatter:
    """A bag of :class:`CostRecord`s with cell-keyed grouping and Pareto queries."""

    records: List[CostRecord] = field(default_factory=list)

    def add(self, record: CostRecord) -> None:
        self.records.append(record)

    def extend(self, records: Sequence[CostRecord]) -> None:
        self.records.extend(records)

    @classmethod
    def from_runs(
        cls, run_dirs: Sequence[str], *, backend: str = "sanafe"
    ) -> "CostScatter":
        """Build a scatter by mining each generated-run directory (skips uncostable runs)."""
        scatter = cls()
        for run_dir in run_dirs:
            record = extract_cost_record_from_run(run_dir, backend=backend)
            if record is not None:
                scatter.add(record)
        return scatter

    def cell_keys(self) -> List[str]:
        """The distinct cell keys present, in stable sorted order."""
        return sorted({r.cell_key for r in self.records})

    def for_cell(self, cell: CertificationCell) -> List[CostRecord]:
        """Every record keyed to ``cell`` (the per-cell scatter)."""
        return [r for r in self.records if r.cell_key == cell.cell_key]

    def by_cell(self) -> Dict[str, List[CostRecord]]:
        """The records grouped by cell key (the cell-keyed scatter)."""
        grouped: Dict[str, List[CostRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.cell_key, []).append(record)
        return grouped

    def pareto_front(
        self, *, cell: Optional[CertificationCell] = None
    ) -> List[CostRecord]:
        """The non-dominated (accuracy↑ × cost↓) records, restricted to ``cell`` when given."""
        pool = self.for_cell(cell) if cell is not None else list(self.records)
        front = [
            candidate
            for candidate in pool
            if not any(
                other is not candidate and _dominates(other, candidate)
                for other in pool
            )
        ]
        return front
