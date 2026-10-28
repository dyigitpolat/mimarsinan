"""[E-3, CL7 follow-up F2] Plane parity on a residual network with projection shortcuts, at the scale CL7 priced it.

CL7's Figure 1 instance (CIFAR ResNet-18 w1/4, BasicBlock [2, 2, 2, 2] with three projection shortcuts, random seeded
weights) at 1/8 of its packed tiles: the exploration plane planned U1 on depth segments, which pack each projection
shortcut beside its block's first conv, while the builder runs it as a pass of its own (23 planned against 26 executed
passes on the 22 nm point before fix_b). The ladder plans the builder's segments since fix_b; F2 asked to check it
again on the integrated head. Here both planes state the same passes, each on the same number of cores, in program
order. Slow: one shape-only census and one integer-program build of the network.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


@pytest.mark.slow
def test_the_ladder_and_the_builder_run_the_same_passes_on_the_cl7_instance():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import scaling_ladder as ladder
    finally:
        sys.path.remove(str(SCRIPTS))
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, resnet18_quarter

    network = ladder.build_network("resnet18_w1q", None, "value")
    chip = ladder.resolved_platform("generic_estimated_22nm", "declared", None)
    census = ladder.shape_census(network, chip)
    tiles = ladder.packed_tiles(census, chip)
    platform = ladder.resolved_platform("generic_estimated_22nm", 0.125, tiles)
    scheduled = ladder.schedule_program(network, census, platform, ladder.program_facts, tiles=tiles)
    planned = [len({h for _s, h in placements}) for placements in scheduled.noc.pass_placements]

    build = ladder.resolved_platform("generic_estimated_22nm", 0.125, tiles,
                                     {"activation_bits": 8, "core_semantics": "mvm"})
    pmap = resolve_platform_mapping_params(list(build["cores"]), allow_coalescing=bool(build["allow_coalescing"]))
    with contextlib.redirect_stdout(io.StringIO()):
        program = build_integer_program(resnet18_quarter(), (3, 32, 32), num_classes=10,
                                        max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons,
                                        allow_coalescing=bool(build["allow_coalescing"]),
                                        weight_bits=int(build["weight_bits"]))
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, build)
    executed = [len(s.hard_core_mapping.cores) for s in hybrid.stages if s.kind == "neural"]
    assert len(planned) == len(executed) == 26
    assert planned == executed
