"""[U-LMP, U-HDA review m-1] On a CUDA-resident program every certificate evaluates on the deployed host and hands
every borrowed module back where it lives.

U-HDA's tree tests run a CUDA-configured pipeline on a host without CUDA, so a twin that evaluated on the pipeline's
device refuses ("No CUDA GPUs are available"); they never place a model, a preprocessor or a host module ON a CUDA
device. The hand-back (``place_on_host`` inside ``borrowed_host_placement``, the builder's ``held_on``) is therefore
pinned only here: the CIFAR fixture's model, a parameter-holding preprocessor and every host op's module live on
``cuda:0`` (the state a GPU session leaves them in), the pipeline names ``cuda:0``, and each case asserts that every
executor forward, host kernel and NF forward ran on the CPU and that every module is where it was. These tests need a
CUDA device and skip by name without one (the reason below); a sura GPU run reads them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.pipelining.core import nf_scm_parity, simulation_factory, spike_count_gate
from unit.pipelining import test_deployed_host_arithmetic as F
from unit.pipelining import test_per_sample_host_twin as cifar

NEEDS_CUDA = ("needs a CUDA device: the hand-back of a CUDA-resident model, preprocessor and host modules "
              "(U-HDA review m-1)")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason=NEEDS_CUDA)
CUDA = "cuda:0"


class _Gain(nn.Module):
    """A preprocessor that holds a parameter (so its placement is observable)."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.gain


def _placement(modules: dict) -> dict:
    return {name: sorted({str(t.device) for t in (*m.parameters(), *m.buffers())}) for name, m in modules.items()}


def _cuda_resident(ir, model):
    """The program's host modules and the model (with a parameter-holding preprocessor) on ``cuda:0``."""
    model.preprocessor = _Gain()
    model.to(CUDA)
    host = {op.name: op.params["module"] for op in ir.get_compute_ops()
            if isinstance(op.params.get("module"), nn.Module)}
    for module in host.values():
        module.to(CUDA)
    assert host, "the fixture's program carries host modules"
    return {"model": model, **{f"host:{name}": module for name, module in host.items()}}


def test_the_spike_count_certificate_on_a_cuda_resident_program_runs_on_the_cpu_and_keeps_placement(monkeypatch):
    model, ir, packed = F._lif_program()
    modules = _cuda_resident(ir, model)
    before = _placement(modules)
    seen = F._record_devices(monkeypatch)
    cert, _traffic = spike_count_gate.run_spike_count_certificate_gate(F._lif_pipeline(CUDA), model, ir, packed)
    assert cert.passed
    assert seen["flow"] and set(seen["flow"]) == {"cpu"}
    assert seen["host"] and set(seen["host"]) == {"cpu"}
    assert _placement(modules) == before


def test_the_streamed_exactness_gate_on_a_cuda_resident_model_runs_on_the_cpu_and_keeps_placement(monkeypatch):
    model, ir, _packed = F._lif_program()
    modules = _cuda_resident(ir, model)
    before = _placement(modules)
    nf_inputs: list = []
    model.register_forward_pre_hook(lambda _m, args: nf_inputs.append(args[0].device.type))
    seen = F._record_devices(monkeypatch)
    nf_scm_parity.assert_streamed_nf_scm_exact_or_raise(F._scm_pipeline(CUDA, 4), model, ir,
                                                        cifar._samples().to(CUDA))
    assert nf_inputs and set(nf_inputs) == {"cpu"}
    assert set(seen["flow"]) <= {"cpu"} and seen["host"] and set(seen["host"]) == {"cpu"}
    assert _placement(modules) == before


def test_the_executor_builder_holds_a_cuda_resident_preprocessor_as_a_cpu_copy_and_keeps_placement(monkeypatch):
    """The build moves no module of the model (the preprocessor is held as a CPU copy); the executor's evaluation, run
    as every pipeline caller runs it (inside ``borrowed_host_placement`` over the model and the program: the census,
    the certificates), evaluates on the CPU and hands every host module back (a host op's module moves to its input's
    device while it runs)."""
    from mimarsinan.chip_simulation.deployed_host import borrowed_host_placement

    model, ir, packed = F._lif_program()
    modules = _cuda_resident(ir, model)
    before = _placement(modules)
    flow = simulation_factory.build_spiking_hybrid_flow(F._lif_pipeline(CUDA), packed, model=model)
    assert {t.device.type for t in flow.parameters()} == {"cpu"}
    assert flow.preprocessor is not model.preprocessor  # a CPU copy; the model's own module never moves
    assert _placement(modules) == before
    seen = F._record_devices(monkeypatch)
    with torch.no_grad(), borrowed_host_placement(model, packed):
        flow(cifar._samples(2).reshape(2, -1))
    assert set(seen["flow"]) == {"cpu"} and seen["host"] and set(seen["host"]) == {"cpu"}
    assert _placement(modules) == before
