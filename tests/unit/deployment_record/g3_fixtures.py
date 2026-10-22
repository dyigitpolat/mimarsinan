"""Hand records, programs and physics shared by the write-law and value-domain tests (G3)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.deployment_record.platform_physics import (
    apply_overrides,
    get_platform_physics,
)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema import (
    ComputeOpRecord,
    PassCarryRecord,
    ScheduleRecord,
    SegmentCoreRecord,
    SegmentRecord,
)

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

from unit.deployment_record.record_fixtures import make_full_record


def quantities(**values: float) -> Quantities:
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def core(index: int, cells: int, *, latency: Optional[int] = 0, params_bytes: Optional[int] = None,
         axons: int = 256, neurons: int = 256) -> SegmentCoreRecord:
    """One hard core whose used rectangle is ``cells`` = cells x 1 (axons_used x neurons_used)."""
    return SegmentCoreRecord(
        core_index=index, axons=axons, neurons=neurons, axons_used=cells, neurons_used=1,
        cells_used=cells, params_bytes=cells if params_bytes is None else params_bytes,
        connectivity_entries=1, static_delay_levels=latency,
    )


def segment(stage: int, seg: int, pass_index: int, cells: Sequence[int], *, resident: bool = False,
            latencies: Optional[Sequence[Optional[int]]] = None,
            marks: Optional[Tuple[int, ...]] = None,
            types: Optional[Sequence[Tuple[int, int]]] = None) -> SegmentRecord:
    """One sealed (segment, pass); ``types`` gives each core's (axons, neurons), 256 x 256 by default."""
    latencies = list(latencies) if latencies is not None else [0] * len(cells)
    types = list(types) if types is not None else [(256, 256)] * len(cells)
    cores = tuple(core(i, c, latency=lat, params_bytes=0 if resident else c, axons=kind[0], neurons=kind[1])
                  for i, (c, lat, kind) in enumerate(zip(cells, latencies, types)))
    return SegmentRecord(
        stage_index=stage, segment_index=seg, pass_index=pass_index,
        pass_reason="initial" if pass_index == 0 else "capacity_overflow",
        programming="resident" if resident else "reprogram", bank_ids=(),
        cores=cores, params_programmed=0 if resident else sum(cells), params_unique=sum(cells),
        params_bytes=sum(c.params_bytes for c in cores), connectivity_entries=len(cores),
        static_latency_levels=len({lat for lat in latencies if lat is not None}),
        resident_core_indices=marks,
    )


def compute(stage: int) -> ComputeOpRecord:
    return ComputeOpRecord(stage_index=stage, name=f"op{stage}", op_type="relu",
                           output_width=4, wall_s_total=None)


def schedule(stages: Sequence[Any], *, carry: Optional[PassCarryRecord] = None) -> ScheduleRecord:
    segments = [s for s in stages if isinstance(s, SegmentRecord)]
    per_segment: Dict[int, int] = {}
    for s in segments:
        per_segment[s.segment_index] = max(per_segment.get(s.segment_index, 0), s.pass_index + 1)
    passes = sum(per_segment.values())
    return ScheduleRecord(
        stages=tuple(stages), pass_count=passes, sync_count=passes - len(per_segment),
        reprogram_passes=sum(s.programming == "reprogram" for s in segments),
        reuse_passes=sum(s.programming == "resident" for s in segments),
        params_reloaded=0, compute_op_count=len(stages) - len(segments), carry=carry,
    )


def record_with(schedule_record: ScheduleRecord, **platform: Any):
    """``make_full_record`` with this schedule and these resolved-platform keys added."""
    base = make_full_record()
    merged = dict(base.identity.platform)
    merged.update(platform)
    return replace(base, schedule=schedule_record,
                   identity=replace(base.identity, platform=merged))


def carry_record(widths: Sequence[int], *, timesteps: int = 32, transfer: str = "collapse",
                 extra_in_trains: Sequence[int] = ()) -> PassCarryRecord:
    """A sealed carry census: one dense crossing per carried slice, plus any extra re-injected trains."""
    per = (timesteps + 7) // 8
    out = sum(w * per for w in widths)
    return PassCarryRecord(
        transfer=transfer, carried_wires=len(widths), carried_bytes=out, peak_live_bytes=out,
        timesteps=timesteps, boundary_out_bytes=out,
        boundary_in_bytes=out + sum(w * per for w in extra_in_trains),
    )


@dataclass(frozen=True)
class HandPassProgram:
    """A ``PassProgram`` twin: per-core committed cells, the whole-pass flag and optional marks."""

    core_cells: Tuple[int, ...]
    resident: bool = False
    schedule_resident_cores: Optional[FrozenSet[int]] = None
    #: (axons, neurons) of each core: the core type a pack placed it on (CL3: G5's PassProgram field).
    core_geometry: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def cores(self) -> int:
        return len(self.core_cells)


def hand_stage(cells: Sequence[int], *, resident: bool = False,
               marks: Optional[FrozenSet[int]] = None) -> SimpleNamespace:
    """A ``HybridStage`` twin carrying the whole-pass flag and, when given, the residency marks."""
    stage = SimpleNamespace(schedule_weights_resident=resident)
    if marks is not None:
        stage.schedule_resident_cores = marks
    stage.cells = tuple(cells)
    return stage


def physics(name: str, **overrides: Dict[str, Any]):
    base = get_platform_physics(name)
    return apply_overrides(base, overrides) if overrides else base


def constant(nominal: float, *, unit: str, low: Optional[float] = None, high: Optional[float] = None,
             note: str = "test constant") -> Dict[str, Any]:
    out: Dict[str, Any] = {"nominal": nominal, "unit": unit, "evidence_kind": "estimated", "note": note}
    if low is not None:
        out["low"] = low
    if high is not None:
        out["high"] = high
    return out


def chain_graph(n=3, width=4):
    """``n`` dependent cores of ``width`` neurons each: a capacity split puts each in its own pass."""
    rng = np.random.default_rng(3)
    nodes = []
    for i in range(n):
        sources = [IRSource(i - 1 if i else -2, j) for j in range(width)] + [IRSource(-3, 0)]
        nodes.append(NeuralCore(id=i, name=f"n{i}", input_sources=np.array(sources, dtype=object),
                                core_matrix=rng.normal(size=(width + 1, width)).astype(np.float32),
                                latency=i, perceptron_index=i))
    return IRGraph(nodes=nodes, output_sources=np.array([IRSource(n - 1, j) for j in range(width)], dtype=object))


def built_chain(count=1):
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True, max_schedule_passes=8))
    return build_hybrid_hard_core_mapping(
        ir_graph=chain_graph(), cores_config=[{"max_axons": 8, "max_neurons": 8, "count": count}],
        strategy=strategy)
