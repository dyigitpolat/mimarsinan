"""The runs a U1 program is checked under, and the identity-equality check every pass projection test uses.

Streamed LIF (VERBATIM boundaries), windowed LIF (per-hop re-timed level stages, COLLAPSE: the deployed lifsync),
cascaded TTFS (COLLAPSE, the transfer every non-streamed run takes, ``run_pass_transfer``), analytical TTFS, and the
synchronized count executor (``lif_execution_synchronized``, COLLAPSE: no deployed run reads it, but the X3a screen
runs it as ``synchronized_lif``, one of the semantics the theorem box covers)."""

from __future__ import annotations

from typing import Optional

import torch

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from spiking_points import point_of_mode
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

from . import vehicles as V

STREAMED_LIF = dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM)
WINDOWED_LIF = dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE)
CASCADED_TTFS = dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded", firing_mode="TTFS",
                     spike_mode="TTFS", pass_transfer=COLLAPSE)
ANALYTICAL_TTFS = dict(spiking_mode="ttfs", firing_mode="TTFS", spike_mode="TTFS", pass_transfer=COLLAPSE)
SYNCHRONIZED_LIF = dict(spiking_mode="lif", cycle_accurate_lif_forward=True, lif_execution_synchronized=True,
                        pass_transfer=COLLAPSE)
ALL_RUNS = (STREAMED_LIF, WINDOWED_LIF, CASCADED_TTFS, ANALYTICAL_TTFS, SYNCHRONIZED_LIF)
RUN_IDS = ("streamed_lif", "windowed_lif", "cascaded_ttfs", "analytical_ttfs", "synchronized_lif")


def flow(mapping, **semantics) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow((V.IN,), mapping, simulation_length=V.T, thresholding_mode="<=", **semantics)


def retimed(semantics) -> bool:
    return semantics is WINDOWED_LIF


def per_cycle(semantics) -> bool:
    """[U-TGAUGE5] Whether the run reads a fused segment cycle by cycle at its latencies, where the per-cycle
    executor holds the window rule: a per-cycle cascade (the mode policy's one predicate) that neither runs level
    units (the windowed run) nor reads whole windows (the synchronized count executor)."""
    policy = policy_for_spiking_mode(point_of_mode(semantics["spiking_mode"], semantics.get("ttfs_cycle_schedule")))
    return (policy.runs_per_cycle_cascade and not retimed(semantics)
            and not semantics.get("lif_execution_synchronized", False))


def mapped(rep, semantics, **kwargs):
    """[U-TGAUGE5] The vehicle's IR for ``semantics`` as Soft Core Mapping builds it where the run reads the window:
    the depth-balancing relays the recipe of every per-cycle cascade arms. The other runs keep the relay-free vehicle,
    whose fused timing (pins, delivery delays) they read without a window."""
    return V.ir_of(rep, relays=per_cycle(semantics), **kwargs)


def planned_budget(ir, count: int, residency_plan, **kwargs) -> int:
    """The smallest core budget at least ``count`` at which ``ir`` plans under ``residency_plan``: the relays add cores,
    so a relayed vehicle may need one or two more than its relay-free twin."""
    from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError

    for budget in range(count, count + 4):
        try:
            V.build(ir, budget, residency_plan, **kwargs)
            return budget
        except ResidencyPlanError:
            continue
    raise AssertionError(f"no plan at {count} to {count + 3} cores")


def assert_refused_at_the_executor(mapping, semantics) -> None:
    """A per-cycle run of a stage holding a live edge at a latency gap other than one is refused by name at the
    executor (the window rule; no mapping armed the relays on the relay-free vehicle)."""
    import pytest

    from mimarsinan.mapping.latency.depth_balancing import UnbalancedFanInError

    x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0))
    with pytest.raises(UnbalancedFanInError, match="no mapping armed"):
        with torch.no_grad():
            flow(mapping, **semantics)(x)


def assert_runs_equal_to_the_identity_run(ir, count: int, semantics, mapping=None, *,
                                          batches: int = 4, **flow_kwargs) -> None:
    """The run of U1's program at ``count`` cores equals the identity run on ``batches`` seeded batches, and the
    vehicle discriminates; windowed LIF builds both programs with per-hop re-timed level stages, as the pipeline
    builds a lifsync program."""
    identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed(semantics))
    scheduled = mapping if mapping is not None else V.build(ir, count, None, retimed_level_stages=retimed(semantics))
    outputs = []
    for batch in range(batches):
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = flow(identity, **semantics, **flow_kwargs)(x)
            assert torch.equal(flow(scheduled, **semantics, **flow_kwargs)(x), want)
        outputs.append(want)
    assert float(torch.cat(outputs).std(0).max()) > 0


def multi_pass(mapping, ir) -> bool:
    """More passes than the identity program has neural stages: a cut happened."""
    identity: Optional[object] = build_identity_hybrid_mapping(ir_graph=ir)
    return len(V.neural_passes(mapping)) > len(V.neural_passes(identity))
