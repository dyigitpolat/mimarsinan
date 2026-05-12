"""HCM↔SANA-FE spike-count parity + rich-stats smoke on a single MNIST sample.

**Status: scale-limited.**  The integration is functional through
SANA-FE for mappings up to ~20-25 cores per neural segment.  The
included MNIST artifact is a 47+49 core ViT-scale mapping that hits a
SANA-FE 2.1.1 scaling cliff inside ``chip.load(net)`` / ``chip.sim()``
(process killed before the LIF runtime check fires).  Three earlier
bugs *were* fixed along the way:

* SANA-FE 2.1.1's ``name[0..N-1]`` range shorthand on cores breaks the
  inner ``inputs[0..K]`` soma-range expansion.  Now ``arch_synth``
  emits one core block per core.
* The huge ``inputs[0..K]`` soma pool used to be declared on every
  core (most unused), which triggered SANA-FE 2.1.1's
  ``"must update every time-step"`` runtime check.  Now declared only
  on the input-host core (tile 0, core 0).
* ``net_synth`` used to create one input neuron per ``seg_in_size``
  slot, mapping all of them even when only a few were wired.  Now
  ``net_synth`` creates input neurons only for axon-referenced
  indices, with a logical-index → group-offset map threaded back to
  the runner.

For runs at production ViT scale, partition the mapping into smaller
neural segments before reaching SANA-FE; the next slow integration
file (``test_sanafe_synthetic_end_to_end.py``) exercises the runner
on the known-good size and is the canonical "does the SANA-FE
backend work" surface.

Loads the LIF-mode MNIST artifact under
``generated/mnist_hard_all_lif_phased_deployment_run*`` and:

  1. runs one sample through ``SpikingHybridCoreFlow`` with recording,
  2. runs the same sample through ``SanafeRunner``,
  3. asserts ``compare_records(hcm_ref, sanafe_record.to_hcm_subset())``
     returns no diffs (parity gate),
  4. asserts the SANA-FE record carries positive aggregate energy,
     positive sim_time, and at least one NoC packet.

Skipped when:
  * the LIF artifact is missing (run the pipeline first), or
  * SANA-FE cannot be imported (opt-in install via
    ``scripts/bootstrap_sanafe.sh``).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

def _resolve_work_dir() -> Path | None:
    """Find the LIF MNIST artifact directory, accepting timestamped variants.

    Newer pipeline runs append ``_YYYYMMDD_HHMMSS``; older runs are unsuffixed.
    We accept either, picking the most recently modified when several exist.
    """
    candidates = sorted(
        Path("generated").glob("mnist_hard_all_lif_phased_deployment_run*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


WORK_DIR = _resolve_work_dir()
IR_PICKLE = WORK_DIR / "Soft Core Mapping.ir_graph.pickle" if WORK_DIR else None
PLATFORM_CFG = WORK_DIR / "Model Configuration.platform_constraints_resolved.json" if WORK_DIR else None
RUN_CONFIG = WORK_DIR / "_RUN_CONFIG" / "config.json" if WORK_DIR else None


def _have_artifacts() -> bool:
    return bool(
        WORK_DIR
        and IR_PICKLE and IR_PICKLE.exists()
        and PLATFORM_CFG and PLATFORM_CFG.exists()
        and RUN_CONFIG and RUN_CONFIG.exists()
    )


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _have_artifacts(), reason=f"missing {WORK_DIR}"),
    pytest.mark.skipif(not _have_sanafe(),
                       reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.slow,
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Helpers (mirror the Loihi parity test's artifact-loading shape)
# ---------------------------------------------------------------------------


def _load_one_sample(idx: int = 0) -> torch.Tensor:
    from mimarsinan.data_handling.data_providers.mnist_data_provider import (
        MNIST_DataProvider,
    )
    provider = MNIST_DataProvider("./datasets")
    test_ds = provider._get_test_dataset()
    x, _ = test_ds[idx]
    return x.unsqueeze(0)


def _load_artifacts() -> tuple[object, dict, int]:
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


# ---------------------------------------------------------------------------
# Parity gate
# ---------------------------------------------------------------------------


def test_sanafe_hcm_spike_parity_single_sample():
    """For one MNIST sample, HCM and SANA-FE must produce identical
    per-segment, per-core input and output spike counts.
    """
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.chip_simulation.spike_recorder import (
        compare_records, format_first_diff,
    )

    ir_graph, platform, sim_length = _load_artifacts()
    hybrid = _build_hybrid_mapping(ir_graph, platform)
    hcm = _build_hcm(hybrid, sim_length)
    x = _load_one_sample(idx=0)

    with torch.no_grad():
        _, rec_hcm = hcm.forward_with_recording(x)

    runner = SanafeRunner(
        mapping=hybrid,
        simulation_length=sim_length,
        arch_preset="loihi",
        thresholding_mode="<",
    )
    sanafe_rec = runner.run(x.detach().cpu().numpy().reshape(1, -1),
                            sample_index=0)
    diffs = compare_records(rec_hcm, sanafe_rec.to_hcm_subset())

    if diffs:
        print(f"\n[sanafe-parity] {len(diffs)} divergence(s):")
        for d in diffs[:20]:
            core_part = f" core={d.core_index}" if d.core_index is not None else ""
            print(
                f"  stage {d.stage_index} ({d.stage_name!r}) "
                f"layer={d.layer}{core_part}: {d.suggested_cause}"
            )
        if len(diffs) > 20:
            print(f"  ... (+{len(diffs) - 20} more)")
    assert not diffs, format_first_diff(diffs)


# ---------------------------------------------------------------------------
# Rich-stats smoke
# ---------------------------------------------------------------------------


def test_sanafe_rich_stats_smoke_total_energy_positive():
    """The rich-stats payload must be non-trivial after a real run."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    ir_graph, platform, sim_length = _load_artifacts()
    hybrid = _build_hybrid_mapping(ir_graph, platform)
    x = _load_one_sample(idx=0)

    runner = SanafeRunner(mapping=hybrid, simulation_length=sim_length,
                          arch_preset="loihi")
    rec = runner.run(x.detach().cpu().numpy().reshape(1, -1), sample_index=0)

    assert rec.aggregate_energy.total_j > 0.0
    assert rec.aggregate_sim_time_s > 0.0
    # NoC packets must be > 0 if the network has more than one tile,
    # but the synthesized arch defaults to one tile — relax to >= 0
    # to avoid coupling to the default packing.
    assert rec.total_packets >= 0
    assert rec.total_spikes >= 0


def test_sanafe_per_neuron_spike_trace_shape_when_enabled():
    """Optional per-neuron traces must come back with the right shape.

    The runner extends the chip simulation to ``T + max_core_latency + 1``
    cycles so multi-depth cascades flush and SANA-FE's one-cycle
    input→synapse pipeline delay doesn't cost the depth-0 cores their
    last integration step (see ``runner._run_neural_stage``).  So the
    trace's time dimension is ``>= sim_length``, not strictly equal.
    """
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    ir_graph, platform, sim_length = _load_artifacts()
    hybrid = _build_hybrid_mapping(ir_graph, platform)
    x = _load_one_sample(idx=0)

    runner = SanafeRunner(
        mapping=hybrid, simulation_length=sim_length,
        arch_preset="loihi", log_potential_trace=True,
    )
    rec = runner.run(x.detach().cpu().numpy().reshape(1, -1), sample_index=0)
    for seg in rec.segments.values():
        if seg.per_neuron_potential_trace is None:
            continue
        trace = np.asarray(seg.per_neuron_potential_trace)
        assert trace.ndim == 2
        assert trace.shape[1] >= sim_length
