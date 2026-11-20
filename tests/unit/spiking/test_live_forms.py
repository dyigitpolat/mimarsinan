"""[U-LIVE] The exactness diagnosis's smallest forms of class A, read element by element by the walk and by the deployed
program.

A dead wired axon (an all-zero weight row) raised its reader's latency in ChipLatency's sweep, the alignment raised an
entry core to serve it, and a shallower live reader of that core read it a cycle late: under streamed LIF it lost a
spike (the gauge tree: 13 of 24 elements apart), and on the window unit's tree the per-cycle executor refuses the
program by name (a live edge at gap 0) wherever the dead read reaches the schedule without the mapping's dead-read pass:
the generator's count sequence and every program built outside Soft Core Mapping. The class is any all-zero row: one
from Weight Quantization (a weight rounded to code zero) and one from the trained module itself, without Weight
Quantization (the check's m-2). Under the one liveness predicate every form reads equal, in both LIF semantics and in
the timing family, built bare, through the mapping and with the relays declared off.
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch
import torch.nn as nn

import unit.spiking.cascade_windows as cw
import unit.spiking.host_gather_forms as hg

from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values
from mimarsinan.mapping.support.host_kernel_evaluation import twin_as_program
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model


def _lin(i, o, w, b):
    layer = nn.Linear(i, o)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(w, dtype=torch.float32).reshape(o, i))
        layer.bias.copy_(torch.as_tensor(b, dtype=torch.float32).reshape(o))
    return nn.Sequential(layer, nn.ReLU())


class Record(nn.Module):
    """The diagnosis's smallest form: E, A over E, C over cat(E, A) with C's weights on A at 0.004 (the 4-bit
    projection rounds them to code 0); the output cat(A, C)."""

    def __init__(self):
        super().__init__()
        self.e = _lin(2, 2, [[0.9, 0.3], [0.2, 0.8]], [0.05, 0.1])
        self.a = _lin(2, 2, [[0.7, 0.2], [0.1, 0.6]], [0.05, 0.0])
        self.c = _lin(4, 1, [[0.5, 0.4, 0.004, 0.004]], [0.02])

    def forward(self, x):
        e = self.e(x)
        a = self.a(e)
        return torch.cat([a, self.c(torch.cat([e, a], dim=1))], dim=1)


class Subsumed(nn.Module):
    """The diagnosis's subsumed form: E on the host, A over E, B over A, C over cat(A, B) with C's weights on B at
    0.004; the output cat(B, C)."""

    def __init__(self):
        super().__init__()
        self.e = _lin(2, 2, [[0.8, 0.3], [0.25, 0.7]], [0.05, 0.05])
        self.a = _lin(2, 2, [[0.7, 0.2], [0.1, 0.6]], [0.05, 0.0])
        self.b = _lin(2, 2, [[0.6, 0.3], [0.2, 0.5]], [0.02, 0.04])
        self.c = _lin(4, 1, [[0.5, 0.4, 0.004, 0.004]], [0.02])

    def forward(self, x):
        a = self.a(self.e(x))
        b = self.b(a)
        return torch.cat([b, self.c(torch.cat([a, b], dim=1))], dim=1)


class Zero(nn.Module):
    """The check's form (m-2): E 3 to 3, A 3 to 2 over E, C 5 to 2 over cat(E, A) whose weights on A are exactly 0.0 in
    the trained module; the output cat(A, C)."""

    def __init__(self):
        super().__init__()
        self.e = _lin(3, 3, [[0.6, 0.2, 0.1], [0.1, 0.7, 0.2], [0.3, 0.1, 0.5]], [0.04, 0.06, 0.02])
        self.a = _lin(3, 2, [[0.5, 0.3, 0.2], [0.2, 0.4, 0.4]], [0.03, 0.05])
        self.c = _lin(5, 2, [[0.4, 0.3, 0.2, 0.0, 0.0], [0.2, 0.5, 0.3, 0.0, 0.0]], [0.02, 0.04])

    def forward(self, x):
        e = self.e(x)
        a = self.a(e)
        return torch.cat([a, self.c(torch.cat([e, a], dim=1))], dim=1)


#: name -> (module, input shape, gauges, encoding placement, Weight Quantization, input seed, samples)
FORMS = {
    "record_wq": (Record, (2,), [1.0, 0.8, 0.9], "offload", True, 8800, 8),
    "subsumed_wq": (Subsumed, (2,), [1.0, 0.8, 0.85, 0.9], "subsume", True, 8801, 8),
    "zero_no_wq": (Zero, (3,), [1.0, 0.7, 0.85], "offload", False, 4242, 16),
}


def _generator_sequence_read(name: str, semantics: str):
    """The form through the generator's count sequence (``host_gather_forms.scm_sequence`` with the declared relays,
    Weight Quantization as the form declares it): the walk in the measurement plane, as the program evaluates its host
    kernels, against the deployed program."""
    factory, shape, gauges, placement, wq, seed, n = FORMS[name]
    module = factory().eval()
    with torch.no_grad():
        n_out = int(module(torch.zeros(2, *shape)).reshape(2, -1).shape[1])
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(module, shape, n_out, device="cpu", encoding_layer_placement=placement).eval()
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, gauges):
        p.set_activation_scale(torch.tensor(g))
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_, ir, hybrid = hg.scm_sequence(flow, perceptrons, wq=wq, retimed=hg.SEMANTICS[semantics][0], relays=True)
    program = hg.Program(name, shape, flow, repr_, ir, hybrid, perceptrons, gauges, wq)
    torch.manual_seed(seed)
    x = 2.0 * torch.rand(n, *shape)

    def walk(prog, xs):
        with torch.no_grad(), measurement_plane():
            return SegmentForwardDriver(prog.repr_, hg.T, hg.SEMANTICS[semantics][2]())(xs)

    ours = twin_as_program(program.hybrid, walk, program, x).double().reshape(n, -1)
    dflow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        deployed = executor_output_values(dflow, dflow(x)).double().reshape(n, -1)
    return ours, deployed, ir


@pytest.mark.parametrize("semantics", sorted(hg.SEMANTICS))
@pytest.mark.parametrize("name", sorted(FORMS))
def test_the_smallest_forms_read_equal_through_the_generators_count_sequence(name, semantics):
    ours, deployed, ir = _generator_sequence_read(name, semantics)
    off = int(((ours - deployed).abs() > 1e-5 * deployed.abs().clamp(min=1.0)).sum())
    assert off == 0, (name, semantics, off, (ours - deployed).abs().max())
    assert not any(getattr(n, "_depth_relay", False) for n in ir.nodes)  # a dead read never asks for a relay


#: An exact-zero form as torch code for the window unit's builders: e (subsumed), E over e, A over E, C over cat(E, A)
#: with every column of A at zero, the output cat(A, C).
OPS = [("lin", 0, 3), ("lin", 1, 3), ("lin", 2, 2), ("cat", [2, 3]), ("lin", 4, 2), ("cat", [3, 5])]
DEAD = {3: [3, 4]}


def _converted(seed: int):
    model = cw.Program(3, OPS, seed, dead=DEAD).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model, (3,), int(model(torch.zeros(2, 3)).shape[1]), device="cpu",
                                   encoding_layer_placement="subsume").eval()
    return flow, (3,)


def _read(family: str, semantics, seed: int, how: str) -> cw.Read:
    flow, shape = _converted(seed)
    torch.manual_seed(11 + seed)
    perceptrons = list(flow.get_perceptrons())
    if family == "count":
        gauges = cw._gauges(flow, perceptrons, seed, 2.0 * torch.rand(16, *shape))
        return cw.count_program(flow, shape, gauges, seed, how, semantics=semantics)
    gauges = cw._gauges(flow, perceptrons, seed, torch.rand(16, *shape))
    return cw.timing_program(flow, shape, gauges, seed, how)


@pytest.mark.parametrize("how", cw.HOWS)
@pytest.mark.parametrize("family,semantics", [("count", "streamed_lif"), ("count", "windowed_lif"), ("timing", None)])
@pytest.mark.parametrize("seed", range(3))
def test_an_exact_zero_row_reads_equal_built_every_way(family, semantics, seed, how):
    """Without Weight Quantization (m-2): built bare (outside the mapping), through the mapping and with the relays
    declared off, in both LIF semantics and in the timing family, the form reads equal with no refusal."""
    read = _read(family, semantics, seed, how)
    assert read.off() == 0, (family, semantics, seed, how, (read.walk - read.deployed).abs().max())
    assert read.relays == 0
