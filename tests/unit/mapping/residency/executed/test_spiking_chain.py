"""The executed plan runs on the spiking executor unchanged: on the seeded multi-level LIF chain, the HCM run of the
executed floor and stateless steps equals the identity (SCM) run and the U1-scheduled run exactly."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage

from . import vehicles as V


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
def test_the_executed_plan_equals_the_identity_and_the_u1_runs(seed, plan):
    ir = V.ir_of(V.chain_repr(seed=seed))
    identity = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
    u1 = V.lif_flow(V.build(ir, 5))
    executed_mapping = V.build(ir, 5, plan)
    assert len(V.neural_passes(executed_mapping)) >= 2
    assert carried_outputs_by_stage(executed_mapping.stages)               # the cut really carries
    executed = V.lif_flow(executed_mapping)
    for batch in range(4):
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(100 * seed + batch))
        with torch.no_grad():
            want = identity(x)
            assert torch.equal(executed(x), want)
            assert torch.equal(u1(x), want)
    assert len({tuple(r.tolist()) for r in want}) >= 3                       # the vehicle discriminates


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
def test_below_the_widest_layer_the_executed_plan_equals_the_identity_run(count, plan):
    """[W03] Tile-granular units: at a budget below the chain's widest layer (3 tiles), where a layer-granular plan
    has none, the executed plan cuts inside a layer and runs equal to the identity (SCM) run at atol 0."""
    ir = V.ir_of(V.chain_repr(seed=0))
    identity = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
    executed_mapping = V.build(ir, count, plan, max_schedule_passes=64)
    passes = V.built_pass_layers(executed_mapping, ir)
    layers = {layer for held in passes for layer in held}
    assert len(passes) >= 6 and any(sum(layer in held for held in passes) >= 2 for layer in layers)   # a split layer
    assert carried_outputs_by_stage(executed_mapping.stages)
    executed = V.lif_flow(executed_mapping)
    for batch in range(4):
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(500 + batch))
        with torch.no_grad():
            assert torch.equal(executed(x), identity(x))
