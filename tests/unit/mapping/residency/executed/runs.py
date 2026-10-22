"""The four deployed runs a U1 program is checked under, and the identity-equality check every pass projection test uses.

Streamed LIF (VERBATIM boundaries), windowed LIF (per-hop re-timed level stages, COLLAPSE: the deployed lifsync),
cascaded TTFS (COLLAPSE, the transfer every non-streamed run takes, ``run_pass_transfer``) and analytical TTFS."""

from __future__ import annotations

from typing import Optional

import torch

from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

from . import vehicles as V

STREAMED_LIF = dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM)
WINDOWED_LIF = dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE)
CASCADED_TTFS = dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded", firing_mode="TTFS",
                     spike_mode="TTFS", pass_transfer=COLLAPSE)
ANALYTICAL_TTFS = dict(spiking_mode="ttfs", firing_mode="TTFS", spike_mode="TTFS", pass_transfer=COLLAPSE)
ALL_RUNS = (STREAMED_LIF, WINDOWED_LIF, CASCADED_TTFS, ANALYTICAL_TTFS)
RUN_IDS = ("streamed_lif", "windowed_lif", "cascaded_ttfs", "analytical_ttfs")


def flow(mapping, **semantics) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow((V.IN,), mapping, simulation_length=V.T, thresholding_mode="<=", **semantics)


def retimed(semantics) -> bool:
    return semantics is WINDOWED_LIF


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
