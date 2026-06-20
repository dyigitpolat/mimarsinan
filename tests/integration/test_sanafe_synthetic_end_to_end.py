"""End-to-end SANA-FE runner test on a synthetic mimarsinan-shaped mapping.

The MNIST artifact (`tests/integration/test_sanafe_hcm_parity.py`) is a
ViT-scale mapping with NoC traffic that triggers SANA-FE 2.1.1's
``"This model must update every time-step"`` runtime check in
``LoihiLifModel::update`` when multiple cores share inputs and use
cross-core spike routing.  We don't fully understand that path yet and
have surfaced the issue separately.

This test pins the runner's end-to-end behaviour on the *known-good*
size: a single-tile, single-core hybrid mapping with input rate-coding
and one neural segment.  When it passes, we know:

* the synthesised arch YAML loads in real SANA-FE,
* the network synthesiser produces a valid SANA-FE Network,
* the runner converts rate-coded input → ``model_attributes["spikes"]``
  → SANA-FE simulation correctly,
* the result is a populated :class:`SanafeRunRecord` with the expected
  spike-count / energy / latency / packet shape.
"""

from __future__ import annotations

import pytest
import numpy as np

from mimarsinan.chip_simulation.sanafe.records import SanafeRunRecord


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _have_sanafe(),
                       reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.slow,
    pytest.mark.integration,
]


def _build_tiny_mapping():
    """One neural stage with one HardCore (4 axons, 3 neurons)."""
    from types import SimpleNamespace
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource

    core = SimpleNamespace(
        axons_per_core=4,
        neurons_per_core=3,
        available_axons=0,
        available_neurons=0,
        threshold=1.0,
        core_matrix=np.asarray([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5],
        ], dtype=np.float32),
        axon_sources=[
            SpikeSource(-1, 0, True, False, False),
            SpikeSource(-1, 1, True, False, False),
            SpikeSource(-1, 2, True, False, False),
            SpikeSource(-1, 3, True, False, False),
        ],
        hardware_bias=None,
        latency=0,
    )
    hcm = SimpleNamespace(cores=[core])

    stage = SimpleNamespace(
        kind="neural", name="tiny", hard_core_mapping=hcm, compute_op=None,
        input_map=[SimpleNamespace(node_id=-2, offset=0, size=4)],
        output_map=[SimpleNamespace(node_id=0, offset=0, size=3)],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    mapping = SimpleNamespace(
        stages=[stage],
        get_neural_segments=lambda: [hcm],
        get_compute_ops=lambda: [],
        output_sources=np.asarray([], dtype=object),
        node_activation_scales={},
        node_input_activation_scales={},
    )
    return mapping


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_runner_end_to_end_on_tiny_mapping():
    """Tiny synthetic mapping runs through the full SANA-FE pipeline."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    mapping = _build_tiny_mapping()

    runner = SanafeRunner(mapping=mapping, simulation_length=32,
                          arch_preset="loihi")
    rates = np.asarray([[1.0, 0.5, 0.25, 0.0]], dtype=np.float32)
    rec = runner.run(rates, sample_index=0)

    assert isinstance(rec, SanafeRunRecord)
    assert rec.sample_index == 0
    assert rec.T == 32
    assert rec.arch_preset == "loihi"
    assert 0 in rec.segments


def test_runner_produces_positive_total_energy():
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    runner = SanafeRunner(mapping=_build_tiny_mapping(), simulation_length=32,
                          arch_preset="loihi")
    rates = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    rec = runner.run(rates, sample_index=0)
    assert rec.aggregate_energy.total_j > 0.0
    # Energy components should sum to approximately the total (numerical noise OK).
    eb = rec.aggregate_energy
    assert abs(eb.components_sum() - eb.total_j) < eb.total_j * 1e-9


def test_runner_produces_positive_sim_time():
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    runner = SanafeRunner(mapping=_build_tiny_mapping(), simulation_length=32,
                          arch_preset="loihi")
    rates = np.asarray([[1.0, 0.5, 0.25, 0.0]], dtype=np.float32)
    rec = runner.run(rates, sample_index=0)
    assert rec.aggregate_sim_time_s > 0.0


def test_runner_records_per_core_output_spikes():
    """Per-core record fields are populated."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    runner = SanafeRunner(mapping=_build_tiny_mapping(), simulation_length=32,
                          arch_preset="loihi")
    rates = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    rec = runner.run(rates, sample_index=0)
    seg = rec.segments[0]
    assert len(seg.per_core) == 1
    pc = seg.per_core[0]
    assert pc.core_index == 0
    assert pc.n_neurons == 3
    assert pc.n_axons_used == 4
    assert pc.output_spike_count.shape == (3,)
    # First two neurons should fire (input rates 1.0 hit weights that drive them
    # past threshold 1.0 every cycle).
    assert pc.output_spike_count[0] > 0
    assert pc.output_spike_count[1] > 0


def test_runner_to_hcm_subset_returns_valid_run_record():
    """The HCM projection produces a RunRecord with matching segment indices."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord
    runner = SanafeRunner(mapping=_build_tiny_mapping(), simulation_length=32,
                          arch_preset="loihi")
    rec = runner.run(np.asarray([[1.0, 0.5, 0.25, 0.0]], dtype=np.float32),
                     sample_index=3)
    sub = rec.to_hcm_subset()
    assert isinstance(sub, RunRecord)
    assert sub.sample_index == 3
    assert sub.T == 32
    assert 0 in sub.segments
    assert len(sub.segments[0].cores) == 1
    # Per-core spike counts flow through verbatim.
    np.testing.assert_array_equal(
        sub.segments[0].cores[0].output_spike_count,
        rec.segments[0].per_core[0].output_spike_count,
    )


# ---------------------------------------------------------------------------
# Real-mimarsinan-mapping smoke at a SANA-FE-friendly scale.
# Uses a subset of the LIF MNIST artifact (top N cores of neural stage 1)
# so we exercise real HardCore weights / biases / thresholds / cross-core
# wiring through the full runner stack — without hitting the SANA-FE 2.1.1
# scaling cliff that the full ViT mapping triggers.
# ---------------------------------------------------------------------------


import json as _json
import pickle as _pickle
from pathlib import Path as _Path


def _have_lif_mnist_artifact() -> _Path | None:
    candidates = sorted(
        _Path("generated").glob("mnist_hard_all_lif_phased_deployment_run*"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for c in candidates:
        if (c / "Soft Core Mapping.ir_graph.pickle").exists() \
           and (c / "Model Configuration.platform_constraints_resolved.json").exists():
            return c
    return None


def _build_subset_hybrid(work_dir: _Path, n_cores: int):
    """Top ``n_cores`` of MNIST stage 1, with cross-core refs > n_cores turned off."""
    import copy
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping, HybridHardCoreMapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.mapping.packing.softcore import HardCoreMapping

    with open(work_dir / "Soft Core Mapping.ir_graph.pickle", "rb") as f:
        ir_graph = _pickle.load(f)
    with open(work_dir / "Model Configuration.platform_constraints_resolved.json") as f:
        platform = _json.load(f)

    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph, cores_config=platform["cores"],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False,
            allow_scheduling=bool(platform.get("allow_scheduling", False)),
            allow_coalescing=False,
        ),
    )
    stage1 = hybrid.stages[1]

    fixed = []
    for c in stage1.hard_core_mapping.cores[:n_cores]:
        c2 = copy.deepcopy(c)
        c2.axon_sources = [
            src if (src.is_off_ or src.is_input_ or src.is_always_on_
                    or src.core_ < n_cores)
            else SpikeSource(0, 0, False, True, False)   # off
            for src in c2.axon_sources
        ]
        fixed.append(c2)

    hcm = HardCoreMapping([])
    hcm.cores = fixed
    hcm.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    stage = copy.deepcopy(stage1)
    stage.hard_core_mapping = hcm
    stage.input_map = [type(s)(node_id=-2, offset=s.offset, size=s.size)
                       for s in stage.input_map]
    return HybridHardCoreMapping(stages=[stage])


@pytest.mark.skipif(_have_lif_mnist_artifact() is None,
                    reason="LIF MNIST artifact missing — run the pipeline first")
def test_runner_handles_real_mnist_subset_with_cross_core_wiring():
    """20-core MNIST subset (cross-core LIFs, real bias/threshold/weights)."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    work_dir = _have_lif_mnist_artifact()
    mapping = _build_subset_hybrid(work_dir, n_cores=20)

    runner = SanafeRunner(mapping=mapping, simulation_length=32,
                          arch_preset="loihi", log_message_trace=False)
    seg_in_size = sum(s.size for s in mapping.stages[0].input_map)
    rates = np.random.RandomState(0).uniform(0.0, 0.3,
                                             size=(1, seg_in_size)).astype(np.float32)
    rec = runner.run(rates, sample_index=0)

    assert rec.aggregate_energy.total_j > 0.0
    assert rec.aggregate_sim_time_s > 0.0
    assert rec.total_spikes > 0
    seg = rec.segments[0]
    assert len(seg.per_core) == 20
    # Every populated core should have a per-core record with the right axon
    # and neuron widths.
    for pc in seg.per_core:
        assert pc.n_neurons > 0
        assert pc.n_axons_used > 0
        assert pc.input_spike_count.shape == (pc.n_axons_used,)
        assert pc.output_spike_count.shape == (pc.n_neurons,)
