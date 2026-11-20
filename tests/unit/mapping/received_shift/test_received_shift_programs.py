"""[U-RSHIFT] The programs the received shift's forms deploy: the record against what each core receives, the banks, and
the walk against the deployed program.

* The field a reader records equals, axon for axon and exactly, the shift the deployed program adds to that reader's
  cores (``node_output_shifts`` placed by each core's own sources): the record is what the chip delivers.
* A reader whose positions read different fields holds one weight bank per distinct field row and bias; no core, axon
  or IR latency is added.
* The walk equals the deployed program: window counts at tolerance 0, the output within float dust (both LIF
  semantics, without and with Weight Quantization at 4 bits), green before and after the unit: the class is a
  conversion error both twins share.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.support.negative_shift import propagate_negative_shifts_to_hybrid, transfer_negative_shifts_to_ir

import unit.spiking.host_gather_forms as hg
from unit.mapping.received_shift import received_shift_forms as rf

FORMS = {
    "padded": lambda g: rf.PadRead(g, "none", 3, 1),
    "join": lambda g: rf.Join(g, 3),
    "two_sources": lambda g: rf.TwoShifted(g, 3),
}


def _program(form: str, *, wq: bool = False, semantics: str = "streamed_lif"):
    model = FORMS[form](torch.Generator().manual_seed(4000))
    flow, repr_, x = rf.build(model, 0)
    rf.policy(flow, x)
    perceptrons = list(flow.get_perceptrons())
    with contextlib.redirect_stdout(io.StringIO()):
        repr_, ir, hybrid = hg.scm_sequence(flow, perceptrons, wq=wq, retimed=hg.SEMANTICS[semantics][0])
        transfer_negative_shifts_to_ir(flow, ir)
        propagate_negative_shifts_to_hybrid(ir, hybrid)
    thetas = [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons]
    return model, flow, rf.conv_readers(repr_)[-1], hg.Program(form, model.shape, flow, repr_, ir, hybrid, perceptrons,
                                                                thetas, wq)


def _reader_cores(program, reader) -> list:
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore) and str(n.name).startswith(f"{reader.name}_pos")]


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_record_is_what_each_core_receives(form):
    _model, _flow, reader, program = _program(form)
    held = reader.perceptron.input_shift.detach().double()
    record = held.reshape(-1)
    table = program.hybrid.node_output_shifts
    worst, compared = 0.0, 0
    for core in _reader_cores(program, reader):
        position = int(core.perceptron_output_column)
        sources = [s for s in core.input_sources.flatten() if int(getattr(s, "node_id", 0)) != -3]
        got = np.zeros(len(sources))
        for i, s in enumerate(sources):
            shift = table.get(int(getattr(s, "node_id", -1)))
            if shift is not None:
                shift = np.asarray(shift).reshape(-1)
                got[i] = shift[0] if shift.size == 1 else shift[int(s.index)]
        if held.dim() == 0:  # one value on every axon (a bias applies it to the padding's too)
            want = np.full(len(sources), float(record))
        elif held.dim() == 1:  # one per input
            want = record.numpy()
        else:  # a field per element of the input, by the reader's own patches
            want = rf.slot_rows(reader, held).numpy()[position]
        worst = max(worst, float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max()))
        compared += len(sources)
    assert compared and worst == 0.0, f"{form}: the record departs from the delivered shift by {worst:.4g} on some axon"


@pytest.mark.parametrize("form", ("padded", "join"))
def test_a_reader_whose_positions_read_different_fields_holds_a_bank_per_distinct_field(form):
    model, _flow, reader, program = _program(form)
    cores = _reader_cores(program, reader)
    banks = {c.weight_bank_id for c in cores}
    shape = rf.input_shape(program.flow, reader, model)
    held = reader.perceptron.input_shift.detach().double()
    record = held.expand(shape) if held.dim() == 0 else held.reshape(shape)
    distinct = int(torch.unique(rf.slot_rows(reader, record), dim=0).shape[0])
    assert len(banks) == distinct, f"{form}: {len(banks)} bank(s) for {distinct} distinct field rows"
    assert len(cores) == int(rf.slot_rows(reader, record).shape[0]), "one core per position, as before"


@pytest.mark.parametrize("wq", (False, True))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_walk_equals_the_deployed_program(form, semantics, wq):
    _model, _flow, _reader, program = _program(form, wq=wq, semantics=semantics)
    mismatches, compared, output = hg.compare(program, semantics, n=8, seed=0)
    assert compared and sum(mismatches.values()) == 0 and output <= 1e-5, (mismatches, compared, output)
