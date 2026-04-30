"""HCM↔Loihi spike-count parity on a single MNIST sample.

Loads the LIF-mode MNIST artifact under
``generated/mnist_hard_all_lif_phased_deployment_run`` (produced by
running the real pipeline through Soft Core Mapping in LIF mode), feeds
**one** test sample through ``SpikingHybridCoreFlow`` (HCM) with
recording, then runs the same per-segment inputs through
``LavaLoihiRunner.run_segments_from_reference`` and asserts the per-core
input/output spike counts agree.

Skipped when:
  * the LIF artifact is missing (run the pipeline first), or
  * Lava cannot be imported (the runner shims out compatibility issues
    when possible, but a missing install is a hard skip).

The harness self-check ``test_hcm_recorder_determinism`` validates the
recorder itself (calling ``forward_with_recording`` twice on the same
sample must yield byte-identical records).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

WORK_DIR = Path("generated/mnist_hard_all_lif_phased_deployment_run")
IR_PICKLE = WORK_DIR / "Soft Core Mapping.ir_graph.pickle"
PLATFORM_CFG = WORK_DIR / "Model Configuration.platform_constraints_resolved.json"
RUN_CONFIG = WORK_DIR / "_RUN_CONFIG" / "config.json"


def _have_artifacts() -> bool:
    return IR_PICKLE.exists() and PLATFORM_CFG.exists() and RUN_CONFIG.exists()


def _have_lava() -> bool:
    try:
        from mimarsinan.chip_simulation.lava_loihi_runner import _probe_lava
        _probe_lava()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _have_artifacts(), reason=f"missing {WORK_DIR}"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_one_sample(idx: int = 0) -> torch.Tensor:
    """Return one MNIST test sample, shape (1, 1, 28, 28), float32."""
    from mimarsinan.data_handling.data_providers.mnist_data_provider import (
        MNIST_DataProvider,
    )
    provider = MNIST_DataProvider("./datasets")
    test_ds = provider._get_test_dataset()
    x, _ = test_ds[idx]
    return x.unsqueeze(0)


def _load_artifacts() -> tuple[object, dict, int]:
    """Load IR graph, platform constraints, and simulation length."""
    with open(IR_PICKLE, "rb") as f:
        ir_graph = pickle.load(f)
    with open(PLATFORM_CFG) as f:
        platform = json.load(f)
    with open(RUN_CONFIG) as f:
        run_cfg = json.load(f)
    sim_length = int(
        run_cfg.get("simulation_steps")
        or run_cfg.get("platform_constraints", {}).get("simulation_steps", 32)
    )
    return ir_graph, platform, sim_length


def _build_hybrid_mapping(ir_graph, platform):
    from mimarsinan.mapping.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )
    return build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=platform["cores"],
        allow_neuron_splitting=bool(platform.get("allow_neuron_splitting", False)),
        allow_scheduling=bool(platform.get("allow_scheduling", False)),
        allow_coalescing=bool(platform.get("allow_coalescing", False)),
    )


def _build_hcm(hybrid_mapping, sim_length: int):
    from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
    return SpikingHybridCoreFlow(
        input_shape=(1, 28, 28),
        hybrid_mapping=hybrid_mapping,
        simulation_length=sim_length,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<",
        spiking_mode="lif",
    ).eval()


def _make_hard_core(
    matrix: np.ndarray,
    axon_sources: list,
    *,
    threshold: float = 1.0,
    hardware_bias: np.ndarray | None = None,
):
    """Construct a minimal occupied ``HardCore`` for synthetic timing tests."""
    from mimarsinan.mapping.softcore_mapping import HardCore

    axons, neurons = matrix.shape
    core = HardCore(axons, neurons)
    core.core_matrix = matrix.astype(np.float32)
    core.axon_sources = axon_sources
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = float(threshold)
    core.input_activation_scale = torch.tensor(1.0)
    core.activation_scale = torch.tensor(1.0)
    core.parameter_scale = torch.tensor(1.0)
    if hardware_bias is not None:
        core.hardware_bias = hardware_bias.astype(np.float32)
    return core


def _make_two_core_hybrid(
    *,
    delayed_core_matrix: np.ndarray,
    delayed_core_sources: list,
    delayed_core_threshold: float = 1.0,
    delayed_core_hardware_bias: np.ndarray | None = None,
):
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.softcore_mapping import HardCoreMapping

    source_core = _make_hard_core(
        np.asarray([[1.0]], dtype=np.float32),
        [SpikeSource(-2, 0, is_input=True)],
    )
    delayed_core = _make_hard_core(
        delayed_core_matrix,
        delayed_core_sources,
        threshold=delayed_core_threshold,
        hardware_bias=delayed_core_hardware_bias,
    )

    segment = HardCoreMapping([])
    segment.cores = [source_core, delayed_core]
    segment.output_sources = np.asarray([SpikeSource(1, 0)], dtype=object)

    stage = HybridStage(
        kind="neural",
        name="synthetic_delayed_segment",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )


@pytest.mark.skipif(not _have_lava(), reason="Lava not importable on this host")
def test_loihi_delayed_core_input_window_matches_hcm():
    """Delayed cores must count segment-input axons over their active window."""
    from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
    from mimarsinan.chip_simulation.spike_recorder import compare_records
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource

    T = 4
    hybrid = _make_two_core_hybrid(
        delayed_core_matrix=np.asarray([[0.0], [1.0]], dtype=np.float32),
        delayed_core_sources=[
            SpikeSource(-2, 0, is_input=True),
            SpikeSource(0, 0),
        ],
    )
    hcm = _build_hcm(hybrid, T)

    with torch.no_grad():
        _, rec_hcm = hcm.forward_with_recording(torch.tensor([[0.25]], dtype=torch.float32))

    delayed_core = rec_hcm.segments[0].cores[1]
    assert delayed_core.core_latency == 1
    assert delayed_core.input_spike_count[0] == 0

    runner = LavaLoihiRunner(pipeline=None, mapping=hybrid, simulation_length=T)
    rec_loihi = runner.run_segments_from_reference(rec_hcm)

    assert not compare_records(rec_hcm, rec_loihi)


@pytest.mark.skipif(not _have_lava(), reason="Lava not importable on this host")
def test_loihi_delayed_hardware_bias_matches_hcm_active_window():
    """Hardware bias must not integrate before a delayed core's active window."""
    from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
    from mimarsinan.chip_simulation.spike_recorder import compare_records
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource

    T = 4
    hybrid = _make_two_core_hybrid(
        delayed_core_matrix=np.asarray([[1.0]], dtype=np.float32),
        delayed_core_sources=[SpikeSource(0, 0)],
        delayed_core_hardware_bias=np.asarray([0.45], dtype=np.float32),
    )
    hcm = _build_hcm(hybrid, T)

    with torch.no_grad():
        _, rec_hcm = hcm.forward_with_recording(torch.tensor([[0.0]], dtype=torch.float32))

    delayed_core = rec_hcm.segments[0].cores[1]
    assert delayed_core.core_latency == 1
    assert delayed_core.output_spike_count[0] == 1

    runner = LavaLoihiRunner(pipeline=None, mapping=hybrid, simulation_length=T)
    rec_loihi = runner.run_segments_from_reference(rec_hcm)

    assert not compare_records(rec_hcm, rec_loihi)


@pytest.mark.skipif(not _have_lava(), reason="Lava not importable on this host")
def test_loihi_duplicate_source_axons_accumulate_weights():
    """Multiple axons reading the same source must sum, not overwrite."""
    from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
    from mimarsinan.chip_simulation.spike_recorder import compare_records
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.softcore_mapping import HardCoreMapping

    T = 4
    segment = HardCoreMapping([])
    segment.cores = [
        _make_hard_core(
            np.asarray([[2.0], [-1.0]], dtype=np.float32),
            [
                SpikeSource(-2, 0, is_input=True),
                SpikeSource(-2, 0, is_input=True),
            ],
            threshold=0.5,
        )
    ]
    segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    stage = HybridStage(
        kind="neural",
        name="duplicate_source_axons",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)],
    )
    hybrid = HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )
    hcm = _build_hcm(hybrid, T)

    with torch.no_grad():
        _, rec_hcm = hcm.forward_with_recording(torch.tensor([[1.0]], dtype=torch.float32))

    runner = LavaLoihiRunner(pipeline=None, mapping=hybrid, simulation_length=T)
    rec_loihi = runner.run_segments_from_reference(rec_hcm)

    assert not compare_records(rec_hcm, rec_loihi)


# ---------------------------------------------------------------------------
# Harness self-check: recorder determinism
# ---------------------------------------------------------------------------


def test_hcm_recorder_determinism():
    """Two consecutive ``forward_with_recording`` calls on the same sample
    must produce byte-identical records.  Catches recorder bugs (off-by-
    one cycle indexing, transposes, accidental float casts).
    """
    ir_graph, platform, sim_length = _load_artifacts()
    hybrid = _build_hybrid_mapping(ir_graph, platform)
    hcm = _build_hcm(hybrid, sim_length)

    x = _load_one_sample(idx=0)

    with torch.no_grad():
        _, rec_a = hcm.forward_with_recording(x)
        _, rec_b = hcm.forward_with_recording(x)

    assert rec_a.T == rec_b.T
    assert set(rec_a.segments) == set(rec_b.segments)
    assert set(rec_a.compute_outputs) == set(rec_b.compute_outputs)

    for sidx in rec_a.segments:
        sa, sb = rec_a.segments[sidx], rec_b.segments[sidx]
        assert np.array_equal(sa.seg_input_spike_count, sb.seg_input_spike_count), (
            f"stage {sidx}: seg_input_spike_count diverged across two HCM runs"
        )
        assert np.array_equal(sa.seg_output_spike_count, sb.seg_output_spike_count), (
            f"stage {sidx}: seg_output_spike_count diverged across two HCM runs"
        )
        assert len(sa.cores) == len(sb.cores)
        for ca, cb in zip(sa.cores, sb.cores):
            assert ca.core_index == cb.core_index
            assert np.array_equal(ca.input_spike_count, cb.input_spike_count), (
                f"stage {sidx} core {ca.core_index}: input_spike_count not deterministic"
            )
            assert np.array_equal(ca.output_spike_count, cb.output_spike_count), (
                f"stage {sidx} core {ca.core_index}: output_spike_count not deterministic"
            )

    for op_id in rec_a.compute_outputs:
        assert np.array_equal(
            rec_a.compute_outputs[op_id], rec_b.compute_outputs[op_id]
        ), f"compute_op {op_id} output not deterministic"


# ---------------------------------------------------------------------------
# Main parity test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_lava(), reason="Lava not importable on this host")
@pytest.mark.slow
def test_loihi_hcm_spike_parity_single_sample():
    """For one MNIST sample, HCM and Loihi must produce identical
    per-segment, per-core input and output spike counts.
    """
    from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
    from mimarsinan.chip_simulation.spike_recorder import (
        compare_records,
        format_first_diff,
    )

    ir_graph, platform, sim_length = _load_artifacts()
    hybrid = _build_hybrid_mapping(ir_graph, platform)

    hcm = _build_hcm(hybrid, sim_length)
    x = _load_one_sample(idx=0)

    with torch.no_grad():
        _, rec_hcm = hcm.forward_with_recording(x)

    print(
        f"\n[parity] HCM record: T={rec_hcm.T}, "
        f"{len(rec_hcm.segments)} neural stages, "
        f"{len(rec_hcm.compute_outputs)} compute outputs"
    )

    runner = LavaLoihiRunner(
        pipeline=None, mapping=hybrid, simulation_length=sim_length,
    )
    rec_loihi = runner.run_segments_from_reference(rec_hcm)

    diffs = compare_records(rec_hcm, rec_loihi)

    if diffs:
        # Print a one-line summary per diff, then the detailed first-diff
        # report.  Useful when there are several so the reader can see
        # the divergence pattern at a glance.
        print(f"\n[parity] {len(diffs)} divergence(s):")
        for d in diffs[:20]:
            core_part = (
                f" core={d.core_index}" if d.core_index is not None else ""
            )
            print(
                f"  stage {d.stage_index} ({d.stage_name!r}) "
                f"layer={d.layer}{core_part}: {d.suggested_cause}"
            )
        if len(diffs) > 20:
            print(f"  ... (+{len(diffs) - 20} more)")

    assert not diffs, format_first_diff(diffs)
