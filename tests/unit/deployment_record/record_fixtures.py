"""Consistent fragment fixtures for deployment-record schema and seal tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import AbstractSet, Optional, Tuple

from mimarsinan.deployment_record.schema import (
    AccuracyReadRecord,
    AccuracyRecord,
    AdaptationRecord,
    Band,
    BankRecord,
    BoundaryTrafficRecord,
    CertificateRecord,
    ComputeOpRecord,
    CrossbarUtilizationRecord,
    DeploymentRecord,
    EnergyRecord,
    EnergyTermRecord,
    FloorplanRecord,
    FtPassWallRecord,
    LatencyDecomposition,
    LayoutStatsRecord,
    ModeledValue,
    NocLinkLoadRecord,
    NocTrafficRecord,
    PlacementRecord,
    Provenance,
    RecordIdentity,
    ScheduleRecord,
    SegmentCoreRecord,
    SegmentRecord,
    SegmentTimingRecord,
    SoftcorePlacementRecord,
    TileRecord,
    TimingRecord,
    TrafficRecord,
    UtilizationRecord,
)

PARAMS_PROGRAMMED_TOTAL = 100
CORES_ALLOCATED = 3
PASS_COUNT = 2


@dataclass(frozen=True)
class PlanView:
    """A SealPlanView stand-in with every predicate defaulting to off."""

    enable_sanafe_simulation: bool = False
    counts_observable: bool = False
    spike_count_gate_armed: bool = False
    nevresim_applies: bool = False
    tuner_hosting_step_names: AbstractSet[str] = frozenset()


def make_provenance(kind: str = "measured", step: str = "Hard Core Mapping") -> Provenance:
    return Provenance(kind=kind, producer="test_producer", step=step, detail="fixture")


def make_identity() -> RecordIdentity:
    return RecordIdentity(
        format_version=1,
        run_dir="run_0001",
        cell_key="lif_streamed@sanafe",
        mode="lif_streamed",
        model_type="mlp",
        model_name="lenet5",
        workload="mnist_t0",
        config_digest="deadbeef" * 8,
        platform={
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
            "weight_bits": 8,
        },
        deployment_options={"schedule_policy": "greedy", "target_tq": 32},
        created_at="2026-08-11T00:00:00+00:00",
    )


def make_core(core_index: int = 0) -> SegmentCoreRecord:
    return SegmentCoreRecord(
        core_index=core_index,
        axons=256,
        neurons=256,
        axons_used=100,
        neurons_used=50,
        cells_used=5000,
        params_bytes=5000,
        connectivity_entries=12,
        static_delay_levels=None,
    )


def make_segment(
    *,
    stage_index: int = 1,
    segment_index: int = 0,
    pass_index: int = 0,
    pass_reason: str = "initial",
    programming: str = "reprogram",
    core_count: int = 2,
    params_programmed: int = PARAMS_PROGRAMMED_TOTAL,
) -> SegmentRecord:
    return SegmentRecord(
        stage_index=stage_index,
        segment_index=segment_index,
        pass_index=pass_index,
        pass_reason=pass_reason,
        programming=programming,
        bank_ids=(0,),
        cores=tuple(make_core(i) for i in range(core_count)),
        params_programmed=params_programmed,
        params_unique=params_programmed,
        params_bytes=params_programmed,
        connectivity_entries=24,
        static_latency_levels=3,
    )


def make_compute_op(stage_index: int = 0) -> ComputeOpRecord:
    return ComputeOpRecord(
        stage_index=stage_index,
        name="maxpool_0",
        op_type="max_pool2d",
        output_width=196,
        wall_s_total=None,
    )


def make_schedule() -> ScheduleRecord:
    return ScheduleRecord(
        stages=(
            make_compute_op(0),
            make_segment(stage_index=1, pass_index=0, core_count=2),
            make_segment(
                stage_index=2,
                pass_index=1,
                pass_reason="capacity_overflow",
                programming="resident",
                core_count=1,
                params_programmed=0,
            ),
        ),
        pass_count=PASS_COUNT,
        sync_count=1,
        reprogram_passes=1,
        reuse_passes=1,
        params_reloaded=PARAMS_PROGRAMMED_TOTAL,
        compute_op_count=1,
    )


def make_softcore_placement() -> SoftcorePlacementRecord:
    return SoftcorePlacementRecord(
        ir_node_id=7,
        segment_index=0,
        pass_index=0,
        hard_core_index=0,
        axon_offset=0,
        neuron_offset=0,
        axons=100,
        neurons=50,
        perceptron_index=3,
        weight_bank_id=0,
        bank_axon_range=(0, 100),
        bank_neuron_range=(0, 50),
        split_group_id=None,
        split_fragment_index=None,
        coalescing_group_id=None,
    )


def make_placement(*, with_floorplan: bool = True) -> PlacementRecord:
    floorplan = (
        FloorplanRecord(mesh_width=2, mesh_height=2, cores_per_tile=4, derivation="derived")
        if with_floorplan
        else None
    )
    return PlacementRecord(
        softcores=(make_softcore_placement(),),
        banks=(BankRecord(bank_id=0, rows=100, cols=50, params=5000, placement_count=2),),
        floorplan=floorplan,
        tiles=(TileRecord(tile_index=0, x=0, y=0, core_indices=(0, 1, 2)),),
    )


def make_crossbar(*, cores_allocated: int = CORES_ALLOCATED) -> CrossbarUtilizationRecord:
    return CrossbarUtilizationRecord(
        cores_allocated=cores_allocated,
        axons_used=300,
        axons_physical=768,
        axon_utilization=300 / 768,
        neurons_used=150,
        neurons_physical=768,
        neuron_utilization=150 / 768,
        cells_used=15000,
        cells_physical=196608,
        cell_occupancy=15000 / 196608,
        unusable_space=0,
        macs=15000,
        weight_bits=8,
        programming_bits=120000,
    )


def make_layout(*, schedule_pass_count: int = PASS_COUNT) -> LayoutStatsRecord:
    zeros = {name: 0.0 for name in (
        "total_wasted_axons_pct", "total_wasted_neurons_pct", "mapped_params_pct",
        "per_core_wasted_axons_pct_min", "per_core_wasted_axons_pct_avg",
        "per_core_wasted_axons_pct_max", "per_core_wasted_neurons_pct_min",
        "per_core_wasted_neurons_pct_avg", "per_core_wasted_neurons_pct_max",
        "per_core_mapped_params_pct_min", "per_core_mapped_params_pct_avg",
        "per_core_mapped_params_pct_max", "segment_latency_min",
        "segment_latency_median", "segment_latency_max",
        "coalescing_frags_per_group_min", "coalescing_frags_per_group_median",
        "coalescing_frags_per_group_max", "splits_per_softcore_min",
        "splits_per_softcore_median", "splits_per_softcore_max", "fragmentation_pct",
    )}
    return LayoutStatsRecord(
        feasible=True,
        total_cores=CORES_ALLOCATED,
        total_softcores=1,
        total_hw_cores=CORES_ALLOCATED,
        coalesced_cores=0,
        split_cores=0,
        neural_segment_count=2,
        residency_class_count=1,
        coalescing_group_count=0,
        split_softcore_count=0,
        schedule_pass_count=schedule_pass_count,
        schedule_sync_count=1,
        max_cores_per_pass=2,
        unused_area_total=0,
        unusable_space_total=0,
        **zeros,
    )


def make_utilization(
    *,
    cores_allocated: int = CORES_ALLOCATED,
    schedule_pass_count: int = PASS_COUNT,
) -> UtilizationRecord:
    return UtilizationRecord(
        crossbar=make_crossbar(cores_allocated=cores_allocated),
        layout=make_layout(schedule_pass_count=schedule_pass_count),
        relay_cores_inserted=1,
    )


def make_traffic(
    *, with_boundaries: bool = True, with_noc: bool = True
) -> TrafficRecord:
    boundaries: Optional[Tuple[BoundaryTrafficRecord, ...]] = (
        (
            BoundaryTrafficRecord(
                node_id=7,
                producing_stage_index=1,
                neurons=50,
                samples=4,
                total_count=1234,
                max_neuron_count=31,
            ),
        )
        if with_boundaries
        else None
    )
    noc = (
        NocTrafficRecord(
            total_packets=1000,
            inter_tile_packets=400,
            intra_tile_packets=500,
            input_path_packets=100,
            cross_tile_connectivity_edges=12,
            mapped_cross_tile_axons=34,
            link_loads=(
                NocLinkLoadRecord(from_x=0, from_y=0, to_x=1, to_y=0, packet_count=400),
            ),
        )
        if with_noc
        else None
    )
    return TrafficRecord(boundaries=boundaries, noc=noc)


def make_band() -> Band:
    return Band(low=1e-4, nominal=1e-3, high=1e-2, basis="barrier cost band")


def make_timing(*, with_per_segment: bool = True) -> TimingRecord:
    per_segment = (
        (
            SegmentTimingRecord(stage_index=1, timesteps_executed=32, sim_time_s=1e-3),
            SegmentTimingRecord(stage_index=2, timesteps_executed=32, sim_time_s=2e-3),
        )
        if with_per_segment
        else ()
    )
    return TimingRecord(
        s_global=32,
        depth=2,
        per_segment=per_segment,
        latency=LatencyDecomposition(
            programming_s=ModeledValue(value=1e-3, band=make_band()),
            compute_steps=64,
            compute_sim_time_s=3e-3 if with_per_segment else None,
            host_ops_s=None,
            host_ops_s_per_pass=None,
            sync_s=ModeledValue(value=1e-3, band=make_band()),
            note="sim_time_s already includes NoC hop latency; no term double-counts it",
        ),
    )


def make_energy() -> EnergyRecord:
    return EnergyRecord(
        total_energy_mj=4.0,
        mj_per_sample=1.0,
        sample_count=4,
        breakdown=(
            EnergyTermRecord(
                name="sanafe_total", mj=4.0, kind="measured", band_mj=None,
                basis="SANA-FE energy trace",
            ),
            EnergyTermRecord(
                name="programming", mj=0.5, kind="modeled", band_mj=(0.1, 1.0),
                basis="DMA per-byte band",
            ),
        ),
        energy_proxy_neuron_steps=4800,
        total_spikes=987,
    )


def make_read(backend: str = "pipeline", kind: str = "measured") -> AccuracyReadRecord:
    return AccuracyReadRecord(
        metric=0.97, backend=backend, samples=128, kind=kind, step="SANA-FE Simulation"
    )


def make_accuracy(*, with_nevresim: bool = True) -> AccuracyRecord:
    reads = (make_read("hcm"),) + ((make_read("nevresim"),) if with_nevresim else ())
    return AccuracyRecord(
        deployed=make_read("pipeline"),
        reads=reads,
        certificates=(
            CertificateRecord(
                name="spike_count_twin",
                backend="nevresim",
                passed=True,
                neuron_windows_compared=1600,
                exact_match_fraction=1.0,
                max_abs_delta=0.0,
                detail="exact under integer accumulation",
            ),
        ),
    )


def make_adaptation() -> AdaptationRecord:
    return AdaptationRecord(
        max_ft_pass_wall_s=12.5,
        ft_pass_walls=(
            FtPassWallRecord(label="lif_adaptation_pass_0", wall_s=12.5),
            FtPassWallRecord(label="lif_adaptation_pass_1", wall_s=9.0),
        ),
    )


def make_full_record() -> DeploymentRecord:
    groups = ("identity", "schedule", "placement", "utilization",
              "accuracy", "timing", "traffic", "energy", "adaptation")
    return DeploymentRecord(
        identity=make_identity(),
        schedule=make_schedule(),
        placement=make_placement(),
        utilization=make_utilization(),
        accuracy=make_accuracy(),
        timing=make_timing(),
        traffic=make_traffic(),
        energy=make_energy(),
        adaptation=make_adaptation(),
        provenance={g: make_provenance() for g in groups},
    )


def segment_with(**overrides) -> SegmentRecord:
    return replace(make_segment(), **overrides)


# ── SANA-FE snapshot fixture (W4.4) ──────────────────────────────────────
# ``SanafeStepReport.to_snapshot_dict()``-shaped, CONSISTENT with
# ``make_schedule``'s census: stage 1 carries 2 cores, stage 2 carries 1.

SNAPSHOT_SEGMENT_CORES = {1: (40, 25), 2: (30,)}
SNAPSHOT_TIMESTEPS = {1: 32, 2: 32}
SNAPSHOT_SIM_TIME_S = {1: 1.5e-3, 2: 2.5e-3}


def _snapshot_segment(stage_index: int) -> dict:
    return {
        "stage_index": stage_index,
        "stage_name": f"segment_{stage_index}",
        "timesteps_executed": SNAPSHOT_TIMESTEPS[stage_index],
        "sim_time_s": SNAPSHOT_SIM_TIME_S[stage_index],
        "per_core": [
            {"core_index": i, "n_neurons": n, "n_axons_used": n + 2}
            for i, n in enumerate(SNAPSHOT_SEGMENT_CORES[stage_index])
        ],
        "inter_tile_packets": 40 * stage_index,
        "intra_tile_packets": 60 * stage_index,
        "input_path_packets": 5,
        "cross_tile_connectivity_edges": 3,
        "mapped_cross_tile_axons": 7,
        "noc_link_load": [
            {"from_x": 0, "from_y": 0, "to_x": 1, "to_y": 0,
             "packet_count": 11 * stage_index},
        ],
        "per_tile": [
            {"tile_index": 0, "cores": [0, 1], "mesh_x": 0, "mesh_y": 0},
        ],
        "arch_geometry": {"width": 2, "height": 2,
                          "tiles_xy": [[0, 0], [0, 1], [1, 0], [1, 1]]},
    }


def make_sanafe_snapshot(*, sample_count: int = 2) -> dict:
    """A ``SanafeStepReport``-shaped snapshot dict over N identical samples."""
    return {
        "arch_preset": "loihi",
        "sample_indices": list(range(sample_count)),
        "aggregate": {
            "sample_count": sample_count,
            "total_energy_j": 4.0e-3,
            "total_energy_mj": 4.0,
            "energy_breakdown_j": {
                "synapse": 1.0e-3, "dendrite": 0.5e-3,
                "soma": 1.5e-3, "network": 1.0e-3, "total": 4.0e-3,
            },
            "max_sim_time_s": 2.5e-3,
            "total_spikes": 987,
            "total_packets": 1234,
        },
        "per_sample": [
            {
                "sample_index": s,
                "T": 32,
                "arch_name": "mimarsinan_loihi_16core",
                "segments": [_snapshot_segment(1), _snapshot_segment(2)],
            }
            for s in range(sample_count)
        ],
    }
