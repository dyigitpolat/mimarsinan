"""[U-TGAUGE3] The U-TGAUGE2 verify's m-2: behavioural tests for the readouts its surviving mutants passed.

U-TGAUGE2's rule (every reader of an output as a value decodes it by its producer's gauge) was pinned at the chip
simulators' readouts by a source-shape check only, at the torch-against-deployed gate by one sample set that ranked a
doubly decoded walk equally, and at the timing teacher by a domain-owning output, where the families' conventions agree.
Each test here runs the reader on a case that tells the right decode from the wrong one: the hybrid nevresim runner and
the Lava runner with their segment runs stubbed (held states at a per-element gauge whose raw ranking differs from the
decoded one), the gate on a second sample set where decoding the walk again flips classes, and the timing teacher on a
program whose output op's timing convention differs from the rate family's. The flat runner's case is in
``test_gauges_placed_by_source_index``.
"""

from __future__ import annotations

import types

import numpy as np
import torch

import unit.spiking.host_gather_forms as hg
import unit.spiking.test_outputs_compared_in_one_unit as tg

from mimarsinan.mapping.ir import IRSource
from mimarsinan.pipelining.core import nf_scm_parity

T = hg.T


def _od():
    from mimarsinan.chip_simulation.hybrid_run import output_decode

    return output_decode


def _held_case(program):
    """Held output states (one sample per row, ``value / gauge``) whose raw ranking differs from the decoded one on
    every row, written per output source into a state buffer, and the decoded classes."""
    gauge = _od().program_output_gauge(program.hybrid)
    order = np.argsort(gauge)
    low, high = int(order[0]), int(order[-1])
    assert gauge[high] > 1.5 * gauge[low]
    held = np.full((3, len(gauge)), 0.05)
    for row, (a, b) in enumerate(((low, high), (low, high), (low, high))):
        held[row, a] = 0.9 - 0.1 * row  # the largest held state: the raw ranking's class
        held[row, b] = 0.8 - 0.1 * row  # at the larger gauge: the decoded value's class
    decoded = np.argmax(held * gauge, axis=1)
    assert bool((decoded != np.argmax(held, axis=1)).all())
    buffer: dict = {}
    for k, src in enumerate(np.asarray(program.hybrid.output_sources).flatten()):
        if isinstance(src, IRSource) and not (src.is_off() or src.is_input() or src.is_always_on()):
            buf = buffer.setdefault(int(src.node_id), np.zeros((3, 64)))
            buf[:, int(src.index)] = held[:, k]
    return buffer, decoded


def _stub_stages(buffer):
    def run(_hybrid, state_buffer, *a, **k):
        state_buffer.update(buffer)

    return run


def test_the_hybrid_nevresim_runner_ranks_the_decoded_outputs(monkeypatch):
    """The verify's VM13 (a runner that decodes at no gauge) passed a source-shape check."""
    from mimarsinan.chip_simulation.simulation_runner import hybrid as runner_module

    program = tg._program("core_per_channel", "streamed_lif")
    buffer, decoded = _held_case(program)
    monkeypatch.setattr(runner_module, "run_hybrid_stages", _stub_stages(buffer))
    monkeypatch.setattr(runner_module, "boundary_normalization_scales", lambda _h: {})
    runner = object.__new__(runner_module.SimulationHybridMixin)
    seen: dict = {}
    for key, value in dict(spiking_mode="lif", simulation_length=T, host_compute_device="cpu", stage_timer=None,
                           membrane_half_step_charge=0.5,
                           test_data=[(np.zeros(10, dtype=np.float32), 0)] * 3).items():
        setattr(runner, key, value)
    runner._prepare_all_segments = lambda _h: []
    runner._evaluate_chip_output = lambda predictions: seen.setdefault("predictions", list(predictions)) and 1.0
    runner._run_hybrid(program.hybrid)
    assert seen["predictions"] == list(decoded)


def test_the_lava_runner_ranks_the_decoded_outputs(monkeypatch):
    """The verify's VM14 (the Lava runner decoding at the mean gauge) passed a source-shape check."""
    from mimarsinan.chip_simulation.lava_loihi import runner as runner_module

    program = tg._program("core_per_channel", "streamed_lif")
    buffer, decoded = _held_case(program)
    monkeypatch.setattr(runner_module, "run_hybrid_stages", _stub_stages(buffer))
    monkeypatch.setattr(runner_module, "boundary_normalization_scales", lambda _h: {})
    runner = object.__new__(runner_module.LavaLoihiRunner)
    runner.mapping, runner.T, runner.pass_transfer = program.hybrid, T, runner_module.COLLAPSE
    runner._profile = types.SimpleNamespace(stages=[], log=lambda: None, total_seconds=0.0)
    runner._load_test_samples = lambda: (np.zeros((3, 10), dtype=np.float32), np.asarray(decoded))
    runner._preprocess = lambda x: x
    reports: list = []
    runner.pipeline = types.SimpleNamespace(reporter=types.SimpleNamespace(report=lambda *a: reports.append(a)))
    assert runner.run() == 1.0


class _Rows(torch.nn.Module):
    """A twin or an executor whose output for a sample is a fixed row (the sample's first feature names it)."""

    def __init__(self, rows, **attrs):
        super().__init__()
        self.rows = torch.as_tensor(rows, dtype=torch.float64)
        for key, value in attrs.items():
            setattr(self, key, value)

    def forward(self, x):
        return self.rows[x[:, 0].long()]


def test_the_torch_against_deployed_gate_ranks_the_walk_as_it_is():
    """The verify's VM04 (the gate decoding the walk's value read again, as if it were a count read) passed on the one
    sample set the gate's test reads. Here the walk's value rows are ranked differently once multiplied by the gauge
    again, and the deployed read holds the same values at the program's gauge: the gate reads one unit."""
    program = tg._program("host_per_channel", "streamed_lif")
    gauge = torch.as_tensor(_od().program_output_gauge(program.hybrid))
    values = torch.zeros(4, len(gauge), dtype=torch.float64)
    for row in range(4):
        values[row, 0], values[row, 1 + row % 2], values[row, len(gauge) - 1] = 1.0, 0.9, 0.5
    counts = values / gauge * T
    again = values / T * gauge
    assert bool((values.argmax(1) != again.argmax(1)).all())  # the case tells the two units apart
    walk = _Rows(values)
    flow = _Rows(counts, hybrid_mapping=program.hybrid, simulation_length=T, spiking_mode="lif")
    x = torch.zeros(4, 10)
    x[:, 0] = torch.arange(4)
    agreement = nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise(walk, flow, x, min_agreement=0.0)
    assert agreement == 1.0


def test_the_timing_teacher_reads_its_output_op_at_the_timing_convention(monkeypatch):
    """The verify's VM19 (the teacher's gauge at the rate family's convention) passed on a domain-owning output, where
    the conventions agree. Here the output is a plain host op: the timing family holds it at its out-scale, the rate
    family at its buffer gauge."""
    import torch.nn as nn

    from mimarsinan.models.spiking.hybrid import identity_flow
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner

    plain = types.SimpleNamespace(id=5, params={"module": nn.Identity()})
    stage = types.SimpleNamespace(kind="compute", compute_op=plain)
    program = types.SimpleNamespace(stages=[stage], output_sources=np.array([IRSource(5, 0), IRSource(5, 1)],
                                                                            dtype=object),
                                    node_activation_scales={5: 2.5}, node_buffer_scales={5: 1.25})
    timing = _od().program_output_gauge(program, timing=True)
    assert not np.allclose(timing, _od().program_output_gauge(program, timing=False))
    monkeypatch.setattr(identity_flow, "build_identity_spiking_flow",
                        lambda *a, **k: types.SimpleNamespace(hybrid_mapping=program))
    host = types.SimpleNamespace(pipeline=types.SimpleNamespace(config={"input_shape": (2,)}), _T=T, _teacher=None,
                                 _thresholding_mode="<=", _map_teacher_to_ir=lambda: None)
    teacher = TTFSCycleAdaptationTuner._build_rung2_teacher(host)
    np.testing.assert_allclose(teacher._output_scales.double().numpy(), timing, rtol=1e-6)
