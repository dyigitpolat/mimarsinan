"""[W02] Position multiplexing on the spiking tier-0 CNN cells at random weights (streamed LIF stream_cnn, synchronized
LIF deep_cnn, streamed LIF LeNet-5), at the cells' own core types: the multiplexed program
equals the SCM identity run and the undeclared (replicated) program, outputs and per-node spike counts, on seeded
batches; each vehicle is scaled by a declared gain until it fires, and the activity is asserted."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts, flow_node_counts
from mimarsinan.mapping.weight_programming import weight_programming_report
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_identity_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)

from unit.mapping.residency.executed import templates as TV

#: (cell, weight gain)
CELLS = [("t0_49", 24.0), ("t0_03", 32.0), ("t0_50", 24.0)]
PASSES = 16384


def _platform(vehicle, multiplexed: bool):
    platform = TV.residency_platform(vehicle, 1.0, None)
    platform["max_schedule_passes"] = PASSES
    if multiplexed:
        platform["position_multiplexing"] = True
    return platform


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cell, gain", CELLS)
def test_the_multiplexed_program_equals_the_identity_and_replicated_runs(cell, gain):
    vehicle = TV.template_vehicle(cell)
    TV.activate(vehicle.ir, gain)
    pipeline, config = vehicle.pipeline(), vehicle.config
    identity = build_spiking_hybrid_flow(
        pipeline, build_identity_mapping_for_pipeline(vehicle.ir, pipeline_config=config))
    replicated_mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, _platform(vehicle, False), pipeline_config=config)
    replicated = build_spiking_hybrid_flow(pipeline, replicated_mapping)
    mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, _platform(vehicle, True), pipeline_config=config)
    neural = [s for s in mapping.stages if s.kind == "neural"]
    assert len(neural) > len([s for s in replicated_mapping.stages if s.kind == "neural"])
    assert 2 * sum(s.schedule_weights_resident for s in neural) >= len(neural)
    programming = weight_programming_report(mapping)
    assert programming.params_programmed == programming.params_unique
    multiplexed = build_spiking_hybrid_flow(pipeline, mapping)
    counts_cell = DeploymentPlan.resolve(config).mode_policy().certification_observable()[0] == "counts"
    for batch in range(2):
        x = torch.rand(4, *vehicle.input_shape, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = identity(x)
            assert torch.equal(multiplexed(x), want)
            assert torch.equal(replicated(x), want)
        assert float(want.std(0).max()) > 0
        if counts_cell:
            cert, report, _ = certify_twin_flow_counts(vehicle.ir, identity, multiplexed, x, backend="hcm",
                                                       discipline="streaming", reference_discipline="streaming")
            assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)
            counts = flow_node_counts(multiplexed, x, discipline="streaming")
            assert sum(float(c.sum()) for c in counts.values()) > 0
