"""The golden's kept node outputs: ``run_golden(..., keep=dict)`` returns the same output codes as without it and hands
back every node output the executor's state buffer materializes, equal at atol 0 (values are codes times steps on
law-carrying nodes)."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs, residual_block_net


def test_kept_node_outputs_cover_every_materialized_node_and_leave_the_outputs_unchanged():
    program = build_integer_program(residual_block_net(), (3, 16, 16), num_classes=10, max_axons=256, max_neurons=32)
    hybrid = program.packed_mapping(axons=256, neurons=32, count=3, passes=64)
    x = integer_inputs(program, 8)
    keep = {}
    codes, steps = run_golden(hybrid, x, keep=keep)
    plain_codes, plain_steps = run_golden(hybrid, x)
    assert np.array_equal(codes, plain_codes) and np.array_equal(steps, plain_steps)
    assert set(keep) == {"codes", "steps", "values"}
    with torch.no_grad():
        _out, state, _flat = ValueHybridCoreFlow(hybrid, dtype=torch.float64, device="cpu").run_retaining_state(x.double())
    assert {k for k in state if k >= 0} == set(keep["values"])
    for node, value in keep["values"].items():
        assert np.array_equal(state[node].numpy(), value)
        if node in keep["codes"]:
            assert np.array_equal(value, keep["codes"][node] * keep["steps"][node])
