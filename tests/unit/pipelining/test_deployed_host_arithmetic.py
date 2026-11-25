"""[U-HDA] Every certificate and every executor of the deployed program runs in the deployed read's arithmetic.

The deployed read (K9) evaluates the deployed program on the CPU: a spiking program by its spiking executor, cores at
float64 on the integer lattice and host kernels at their deployment dtype; a value program at float64. A certificate
certifies the arithmetic of the row it is printed on only where both of its twins evaluate there too. On a pipeline
whose training device is CUDA the certificates evaluated the deployed program's host kernels on CUDA (the contract's
host-op evaluator was the pipeline's device), and a CUDA device holds neither float32 host convolutions nor float64
reductions bit for bit: the tuned C5 B0 program classes 7 of 5,000 E01 images differently there (U-CEN's records).

These tests pin every certificate, both twins of each, and every executor to the deployed host on a pipeline that
names a CUDA device. The host running them has no CUDA device visible (the suite's scopes hide it), so a twin that
evaluated where the pipeline names refuses here ("No CUDA GPUs are available"), and a twin on the deployed host runs:

- the SSOT (``chip_simulation/deployed_host.py``) names the CPU whatever device the pipeline names, and the census
  reads it;
- the contract's host-op evaluator and every chip runner's (nevresim, Lava) are the deployed host's;
- the spike-count certificate (K7: the identity and packed twins), the NF twin of every parity gate, the HCM twin of
  nevresim's certificate (K8), the HCM references of the Loihi, SANA-FE and both ODIN steps, the value certificates
  (K4, K5), the event census and the reference builders evaluate every forward and every host kernel on the CPU, and
  their verdicts and records equal the same run on a CPU pipeline;
- each certificate prints its host device and dtype (``[HostArithmetic]``);
- the placement of every module an evaluation borrows is handed back (``borrowed_host_placement``), and a module the
  executor reads from the model is held as a copy where it lives elsewhere (``held_on``).

A CUDA-resident model (its placement handed back on exit) needs a CUDA device: that reading is the unit's GPU probe on a
server GPU (final_units/uhda), not a test here, because a test that skips without CUDA is no reading in the suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, TinyDataProvider, default_config

from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.host_kernel_evaluation import PER_SAMPLE, declare_host_kernel_evaluation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core import nf_scm_parity, simulation_factory, spike_count_gate
from mimarsinan.pipelining.core.gates import value_gates
from mimarsinan.torch_mapping.converter import convert_torch_model

from unit.pipelining import test_per_sample_host_twin as cifar

CUDA_NAMES = ("cuda", "cuda:0", "cuda:1")


# ---------------------------------------------------------------------------
# recorders: where every forward of an executor and every host kernel ran
# ---------------------------------------------------------------------------


def _record_devices(monkeypatch) -> dict:
    seen: dict = {"flow": [], "host": []}
    forward = SpikingHybridCoreFlow.forward
    exec_module = ComputeOp._exec_module

    def flow_forward(self, x):
        seen["flow"].append(x.device.type)
        return forward(self, x)

    def host_kernel(self, x):
        seen["host"].append(x.device.type)
        return exec_module(self, x)

    monkeypatch.setattr(SpikingHybridCoreFlow, "forward", flow_forward)
    monkeypatch.setattr(ComputeOp, "_exec_module", host_kernel)
    return seen


def _host_lines(out: str) -> list:
    return [ln for ln in out.splitlines() if ln.startswith("[HostArithmetic]")]


# ---------------------------------------------------------------------------
# the CIFAR fixture's program on a pipeline: an fp32 host convolution stem, a host residual, an fp32 host classifier
# ---------------------------------------------------------------------------


class _Provider(TinyDataProvider):
    def __init__(self):
        super().__init__(input_shape=cifar.SHAPE, num_classes=cifar.N_CLASSES, size=8)
        self._ds.data = cifar._samples(8)
        self._ds.targets = torch.arange(8) % cifar.N_CLASSES


class _Factory:
    def __init__(self):
        self._provider = _Provider()

    def create(self):
        return self._provider


def _lif_config(device: str, weight_bits) -> dict:
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": cifar.T,
                "input_shape": cifar.SHAPE, "num_classes": cifar.N_CLASSES, "device": device, "num_workers": 0,
                "weight_quantization": weight_bits is not None, "spike_count_parity_samples": 4})
    return cfg


def _lif_pipeline(device: str, weight_bits=4):
    pipeline = MockPipeline(config=_lif_config(device, weight_bits), data_provider_factory=_Factory())
    return pipeline


def _lif_program(weight_bits=4, per_sample=True):
    flow, ir = cifar._program(weight_bits)
    if per_sample:
        declare_host_kernel_evaluation(ir, PER_SAMPLE)
    packed = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 400}])
    return cifar._NF(flow), ir, packed


# ---------------------------------------------------------------------------
# the SSOT
# ---------------------------------------------------------------------------


def test_the_deployed_host_is_the_cpu_whatever_device_the_pipeline_names():
    from mimarsinan.chip_simulation.deployed_host import (
        DEPLOYED_HOST_DEVICE,
        deployed_host_device,
    )
    from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC  # [U-DAR] the record
    from mimarsinan.pipelining.core.census import arithmetic as census

    assert DEPLOYED_HOST_DEVICE == "cpu" and VALUE_ARITHMETIC.core_dtype is torch.float64
    for asked in ("cpu", torch.device("cpu"), *CUDA_NAMES, torch.device("cuda", 2), None):
        assert deployed_host_device(asked) == "cpu", asked
    # the census reads the one rule and the one dtype (the record's: U-DAR)
    pipe = SimpleNamespace(config={"device": "cuda"})
    assert census.value_census_arithmetic(pipe, "c").dtype == str(VALUE_ARITHMETIC.core_dtype).removeprefix("torch.")
    assert census.value_census_arithmetic(pipe, "c").device == DEPLOYED_HOST_DEVICE
    assert all(census.census_device(asked) == deployed_host_device(asked) for asked in ("cpu", *CUDA_NAMES, None))


def test_the_arithmetic_line_names_the_device_the_dtypes_and_the_move():
    from mimarsinan.chip_simulation.deployed_host import spiking_host_arithmetic, value_host_arithmetic

    _model, ir, packed = _lif_program()
    moved = spiking_host_arithmetic("K7", "cuda:0", ir, packed)
    assert moved.device == "cpu" and moved.moved and moved.dtype == "cores float64, host kernels float32"
    assert moved.line().startswith("[HostArithmetic] K7: device=cpu dtype=cores float64, host kernels float32; "
                                   "pipeline device cuda:0")
    kept = value_host_arithmetic("K5", "cpu")
    assert kept.line() == "[HostArithmetic] K5: device=cpu dtype=float64" and not kept.moved


# ---------------------------------------------------------------------------
# the host-op evaluator of the contract and of every chip runner
# ---------------------------------------------------------------------------


def _contract_config(device):
    return {"simulation_steps": 4, "spiking_mode": "lif", "firing_mode": "Default", "thresholding_mode": "<",
            "spike_generation_mode": "Uniform", "device": device}


@pytest.mark.parametrize("device", [*CUDA_NAMES, "cpu"])
def test_the_contract_s_host_op_evaluator_is_the_deployed_host(device):
    """SANA-FE, both ODIN twins and every backend that reads the contract re-derive host ops here."""
    assert SpikingDeploymentContract.from_pipeline_config(_contract_config(device)).host_compute_device == "cpu"


@pytest.mark.parametrize("device", CUDA_NAMES)
def test_nevresim_s_runner_evaluates_host_kernels_on_the_deployed_host(device):
    from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

    config = default_config()
    config.update(device=device, input_size=64)
    runner = SimulationRunner(MockPipeline(config=config), mapping=None, simulation_length=4)
    assert runner.host_compute_device == "cpu"


@pytest.mark.parametrize("device", CUDA_NAMES)
def test_lava_s_runner_evaluates_host_kernels_on_the_deployed_host(device):
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner

    config = default_config()
    config["device"] = device
    behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<",
                                    spike_generation_mode="Uniform")
    _model, _ir, packed = _lif_program()
    runner = LavaLoihiRunner(mapping=packed, simulation_length=4, behavior=behavior, pipeline=MockPipeline(config=config))
    assert runner.device == "cpu"


# ---------------------------------------------------------------------------
# K7: the spike-count certificate's identity and packed twins
# ---------------------------------------------------------------------------


def _k7(device, monkeypatch=None):
    model, ir, packed = _lif_program()
    seen = _record_devices(monkeypatch) if monkeypatch is not None else None
    cert, traffic = spike_count_gate.run_spike_count_certificate_gate(_lif_pipeline(device), model, ir, packed)
    return cert, traffic, seen, model


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_the_spike_count_certificate_runs_both_twins_on_the_deployed_host(device, monkeypatch, capsys):
    on_cpu, cpu_traffic, _, _ = _k7("cpu")
    capsys.readouterr()
    cert, traffic, seen, model = _k7(device, monkeypatch)
    assert cert.passed and cert.summary() == on_cpu.summary() and traffic == cpu_traffic
    assert seen["flow"] and set(seen["flow"]) == {"cpu"}
    assert seen["host"] and set(seen["host"]) == {"cpu"}
    assert {p.device.type for p in model.parameters()} == {"cpu"}
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "spike-count certificate (K7" in lines[0]
    assert f"device=cpu dtype=cores float64, host kernels float32; pipeline device {device}" in lines[0]


def test_the_spike_count_certificate_s_samples_are_on_the_deployed_host():
    model, _ir, _packed = _lif_program()
    samples = spike_count_gate._certificate_samples(_lif_pipeline("cuda:0"), model, 3)
    assert samples.device.type == "cpu" and tuple(samples.shape) == (3, *cifar.SHAPE)
    assert torch.equal(samples, _Provider()._ds.data[:3])


# ---------------------------------------------------------------------------
# the NF twin of every parity gate
# ---------------------------------------------------------------------------


def _scm_pipeline(device, weight_bits):
    pipeline = cifar._pipeline(weight_bits)
    pipeline.config["device"] = device
    return pipeline


@pytest.mark.parametrize("weight_bits", [None, 4], ids=["float_weights", "wq4"])
def test_the_streamed_exactness_gate_runs_both_twins_on_the_deployed_host(weight_bits, monkeypatch, capsys):
    flow, ir = cifar._program(weight_bits)
    declare_host_kernel_evaluation(ir, PER_SAMPLE)
    model = cifar._NF(flow)
    nf_inputs: list = []
    model.register_forward_pre_hook(lambda _m, args: nf_inputs.append(args[0].device.type))
    seen = _record_devices(monkeypatch)
    nf_scm_parity.assert_streamed_nf_scm_exact_or_raise(_scm_pipeline("cuda:0", weight_bits), model, ir,
                                                        cifar._samples())
    assert nf_inputs and set(nf_inputs) == {"cpu"}
    assert set(seen["flow"]) == {"cpu"} and seen["host"] and set(seen["host"]) == {"cpu"}
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "streamed NF↔SCM exactness" in lines[0] and "device=cpu" in lines[0]
    assert "pipeline device cuda:0" in lines[0]


def test_the_torch_versus_deployed_gate_runs_both_twins_on_the_deployed_host(monkeypatch, capsys):
    flow, ir = cifar._program(4)
    declare_host_kernel_evaluation(ir, PER_SAMPLE)
    model = cifar._NF(flow)
    nf_inputs: list = []
    model.register_forward_pre_hook(lambda _m, args: nf_inputs.append(args[0].device.type))
    deployed = simulation_factory.build_spiking_hybrid_flow(
        _lif_pipeline("cuda:0"), build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 400}]))
    seen = _record_devices(monkeypatch)
    agreement = nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise(model, deployed, cifar._samples(),
                                                                           min_agreement=0.5)
    assert 0.5 <= agreement <= 1.0
    assert nf_inputs and set(nf_inputs) == {"cpu"} and set(seen["flow"]) == {"cpu"} and set(seen["host"]) == {"cpu"}
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "torch-model↔deployed-sim parity" in lines[0] and "device=cpu" in lines[0]


def test_the_per_neuron_and_cascaded_gates_print_the_deployed_host(monkeypatch, capsys):
    from unit.pipelining.pipeline_steps import test_nf_scm_parity_gate as toy

    model, ir = toy._build_aligned_toy()
    pipeline = toy._pipeline()
    pipeline.config["device"] = "cuda"
    torch.manual_seed(1)
    assert nf_scm_parity.assert_nf_scm_parity_or_raise(pipeline, model, ir,
                                                       torch.rand(3, 8, dtype=torch.float64)) == 0.0
    cascaded = toy._pipeline("cascaded", spiking_mode="ttfs_cycle_based")
    cascaded.config["device"] = "cuda:1"
    echo = toy.TestCascadedDecisionAgreement()
    linear = echo._setup(monkeypatch, flip=False)
    assert nf_scm_parity.assert_cascaded_nf_scm_agreement_or_raise(
        cascaded, linear, ir_graph=object(), samples=torch.rand(16, 8, dtype=torch.float64), min_agreement=0.85) == 1.0
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert "NF↔SCM per-neuron parity" in lines[0] and "pipeline device cuda" in lines[0]
    assert "cascaded NF↔SCM decision agreement" in lines[1] and "pipeline device cuda:1" in lines[1]
    assert all("device=cpu" in ln for ln in lines)


# ---------------------------------------------------------------------------
# K8 and the chip backends' HCM references
# ---------------------------------------------------------------------------


def _streaming_counts(pipeline, packed, samples) -> list:
    captured: list = []
    flow = simulation_factory.build_spiking_hybrid_flow(pipeline, packed, model=None)
    flow.stage_count_recorder = lambda stage, counts: captured.append((stage, counts.detach().cpu().numpy()))
    flow.lif_execution_synchronized = False
    from mimarsinan.models.nn.lif_kernels import measurement_plane

    with measurement_plane(), torch.no_grad():
        flow(samples)
    return captured


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_nevresim_s_certificate_runs_its_hcm_twin_on_the_deployed_host(device, monkeypatch, capsys):
    from mimarsinan.pipelining.pipeline_steps.verification.simulation_step import _certify_nevresim_counts

    _model, _ir, packed = _lif_program()
    samples = cifar._samples(3)
    captured = _streaming_counts(_lif_pipeline("cpu"), packed, samples)
    seen = _record_devices(monkeypatch)
    cert = _certify_nevresim_counts(pipeline=_lif_pipeline(device), mapping=packed, captured=captured,
                                    samples=samples)
    assert cert.exact_match_fraction == 1.0 and cert.max_abs_delta == 0
    assert set(seen["flow"]) == {"cpu"} and seen["host"] and set(seen["host"]) == {"cpu"}
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "nevresim spike-count certificate (K8" in lines[0] and "device=cpu" in lines[0]


@pytest.mark.parametrize("device", CUDA_NAMES)
def test_every_hcm_reference_builder_evaluates_on_the_deployed_host(device, monkeypatch):
    _model, _ir, packed = _lif_program()
    pipeline = _lif_pipeline(device)
    sample = cifar._samples(1)
    seen = _record_devices(monkeypatch)
    flow, record = simulation_factory.record_hcm_reference(pipeline, packed, sample)
    reference = simulation_factory.hcm_reference_flow(pipeline, packed)
    again = simulation_factory.record_on_hcm_flow(reference, sample, sample_index=0, device="cpu")
    x = simulation_factory.preprocess_hybrid_sample(pipeline, packed, sample)
    assert x.dtype == np.float64 and x.shape == (1, int(np.prod(cifar.SHAPE)))
    assert set(seen["flow"]) == {"cpu"} and set(seen["host"]) == {"cpu"}
    assert flow.preprocessor is not None and record.segments.keys() == again.segments.keys()


def _loihi_step(monkeypatch, device):
    from unit.pipelining.pipeline_steps import test_loihi_simulation_step as loihi_t
    import mimarsinan.pipelining.pipeline_steps.verification.loihi_simulation_step as loihi_step

    step, pipeline, calls = loihi_t._prepare_step(monkeypatch)
    pipeline.config["device"] = device
    fake = loihi_step.record_hcm_reference
    devices: list = []

    def recording(pipeline_, mapping, sample, sample_index=0, device=None):
        devices.append(device)
        return fake(pipeline_, mapping, sample, sample_index=sample_index, device=device)

    monkeypatch.setattr(loihi_step, "record_hcm_reference", recording)
    return step, devices, calls


def test_the_loihi_step_s_hcm_reference_runs_on_the_deployed_host(monkeypatch, capsys):
    step, devices, calls = _loihi_step(monkeypatch, "cuda:0")
    step.run()
    assert devices == ["cpu"] and calls["segments_from_reference"] == 1
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "Loihi Simulation spike parity" in lines[0] and "pipeline device cuda:0" in lines[0]


def test_the_sanafe_step_s_hcm_reference_and_host_kernels_run_on_the_deployed_host(monkeypatch, capsys):
    from unit.pipelining.pipeline_steps import test_sanafe_simulation_step as sanafe_t
    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as sanafe_step

    step, pipeline, calls = sanafe_t._prepare_step(monkeypatch, sample_count=2)
    pipeline.config["device"] = "cuda:1"
    fake = sanafe_step.record_hcm_reference
    devices: list = []

    def recording(pipeline_, mapping, sample, sample_index=0, device=None):
        devices.append(device)
        return fake(pipeline_, mapping, sample, sample_index=sample_index, device=device)

    monkeypatch.setattr(sanafe_step, "record_hcm_reference", recording)
    step.run()
    assert devices == ["cpu", "cpu"]
    assert [init["contract"].host_compute_device for init in calls["runner_inits"]] == ["cpu", "cpu"]
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "SANA-FE Simulation spike parity" in lines[0] and "pipeline device cuda:1" in lines[0]


def test_the_odin_fpga_step_certifies_the_device_against_a_reference_on_the_deployed_host(monkeypatch, capsys):
    from integration.odin_fpga_harness import prepare_step, two_core_mapping
    from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_fpga_simulation_step import (
        OdinFpgaDeploymentStep,
    )
    from unit.pipelining.pipeline_steps.test_odin_fpga_step import TwinTransport

    certificates = {}
    for device in ("cpu", "cuda:0"):
        seen = _record_devices(monkeypatch)
        pipeline, step = prepare_step(monkeypatch, OdinFpgaDeploymentStep, transport=TwinTransport(two_core_mapping()),
                                      config_overrides={"device": device})
        step.process()
        certificates[device] = [
            {k: [np.asarray(c.output_spike_count).tolist() for c in seg.cores]
             for k, seg in r.to_hcm_subset().segments.items()} for r in step.deployed_records]
        assert certificates[device] and seen["flow"] and set(seen["flow"]) == {"cpu"}
        monkeypatch.undo()
    assert certificates["cuda:0"] == certificates["cpu"]
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 2 and all("ODIN FPGA spike parity" in ln and "device=cpu" in ln for ln in lines)


def test_the_odin_hacc_bundle_on_a_cuda_pipeline_is_the_cpu_pipeline_s(monkeypatch, tmp_path, capsys):
    from integration.odin_hacc_harness import prepare_step
    from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_hacc_deployment_step import (
        OdinHaccDeploymentStep,
    )

    documents = {}
    for device in ("cpu", "cuda:0"):
        seen = _record_devices(monkeypatch)
        _pipeline, step = prepare_step(monkeypatch, OdinHaccDeploymentStep, working_directory=str(tmp_path / device),
                                       config_overrides={"device": device})
        step.process()
        documents[device] = step.document
        assert seen["flow"] and set(seen["flow"]) == {"cpu"}
        monkeypatch.undo()
    assert documents["cuda:0"] == documents["cpu"]
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 2 and all("ODIN HACC bundle freeze" in ln and "device=cpu" in ln for ln in lines)


# ---------------------------------------------------------------------------
# the event census and the membrane-readout diagnostic
# ---------------------------------------------------------------------------


def test_the_event_census_on_a_cuda_pipeline_is_the_cpu_pipeline_s(tmp_path, monkeypatch, capsys):
    from unit.pipelining.pipeline_steps import test_event_census_step as census_t

    from mimarsinan.pipelining.pipeline_steps import EventCensusStep

    censuses = {}
    for device in ("cpu", "cuda:0"):
        pipeline, _mapping = census_t._census_pipeline(tmp_path / device, samples=3)
        pipeline.config["device"] = device
        step = EventCensusStep(pipeline)
        pipeline.prepare_step(step)
        seen = _record_devices(monkeypatch)
        step.run()
        censuses[device] = pipeline.cache[f"{step.name}.event_census"]
        assert seen["flow"] and set(seen["flow"]) == {"cpu"}
        monkeypatch.undo()
    assert censuses["cuda:0"] == censuses["cpu"]
    lines = _host_lines(capsys.readouterr().out)
    assert len(lines) == 2 and all("Event Census" in ln and "device=cpu" in ln for ln in lines)


# ---------------------------------------------------------------------------
# K4 and K5: the value certificates
# ---------------------------------------------------------------------------


def _value_pipeline(device: str):
    config = default_config()
    config.update(core_semantics="mvm", device=device, input_shape=(4,), num_classes=2, value_parity_samples=3)
    provider = TinyDataProvider(input_shape=(4,), num_classes=2, size=6)
    generator = torch.Generator().manual_seed(5)
    provider._ds.data = torch.rand(6, 4, generator=generator)
    factory = SimpleNamespace(create=lambda: provider)
    pipeline = MockPipeline(config=config, data_provider_factory=factory)
    return pipeline


def _value_program():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    model = convert_torch_model(net, (4,), 2, device="cpu", packaging=MVM_PACKAGING).eval()
    mapper = model.get_mapper_repr()
    mapper.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64).map(mapper)
    packed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64,
                                                                        "count": 9}])
    return model, ir, packed


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_the_value_certificates_run_in_the_deployed_arithmetic_and_print_it(device, capsys):
    model, ir, packed = _value_program()
    value_gates.run_model_value_parity_gate(_value_pipeline(device), model, ir)
    twin = value_gates.run_value_twin_certificate_gate(_value_pipeline(device), model, ir, packed)
    assert twin is not None and twin.passed
    out = capsys.readouterr().out
    assert "[ValueParityGate] PASS" in out and "[ValueTwinCertificate] PASS" in out
    lines = _host_lines(out)
    assert len(lines) == 2
    assert "value R-edge (K4" in lines[0] and "value twin certificate (K5" in lines[1]
    assert all(f"device=cpu dtype=float64; pipeline device {device}" in ln for ln in lines)
    assert {p.device.type for p in model.parameters()} == {"cpu"}


# ---------------------------------------------------------------------------
# placement: every borrowed module handed back; a model's module held as a copy where it lives elsewhere
# ---------------------------------------------------------------------------


def test_a_borrowed_module_is_handed_back_where_it_was():
    from mimarsinan.chip_simulation.deployed_host import borrowed_host_placement

    model = nn.Linear(3, 2)
    host_module = nn.Linear(2, 2)
    program = SimpleNamespace(stages=[SimpleNamespace(compute_op=SimpleNamespace(params={"module": host_module}))])
    originals = [model.weight, model.weight.data_ptr(), host_module.weight.data_ptr()]
    with borrowed_host_placement(model, program, moved=True):
        model.to("meta")
        host_module.to("meta")
        assert model.weight.device.type == "meta" and host_module.weight.device.type == "meta"
    assert model.weight is originals[0] and model.weight.data_ptr() == originals[1]
    assert host_module.weight.data_ptr() == originals[2]
    assert model.weight.device.type == "cpu" and host_module.weight.device.type == "cpu"


def test_nothing_on_the_deployed_host_is_restored():
    """On a CPU pipeline an evaluation is left exactly as it was: no module living on the host is restored."""
    from mimarsinan.chip_simulation.deployed_host import borrowed_host_placement

    model = nn.Linear(3, 2)
    replaced = torch.nn.Parameter(torch.zeros(2, 3))
    with borrowed_host_placement(model):
        model.weight = replaced
    assert model.weight is replaced


def test_the_executor_holds_a_model_s_module_living_elsewhere_as_a_copy():
    """A module already on the executor's device (or holding no tensor) is held as it is; one living on another device
    is held as a copy there, and the model's own module never moves (``meta`` stands in for the other device)."""
    from mimarsinan.chip_simulation.deployed_host import held_on

    gain = nn.Linear(4, 4)
    identity = nn.Identity()
    assert held_on(gain, "cpu") is gain and held_on(identity, "meta") is identity and held_on(None, "cpu") is None
    weight = gain.weight.data_ptr()
    copied = held_on(gain, "meta")
    assert copied is not gain and copied.weight.device.type == "meta"
    assert gain.weight.data_ptr() == weight and gain.weight.device.type == "cpu"


@pytest.mark.parametrize("device", CUDA_NAMES)
def test_the_executor_is_built_on_the_deployed_host_whatever_device_the_pipeline_names(device, monkeypatch):
    _model, _ir, packed = _lif_program()
    model = cifar._NF(cifar._program(4)[0])
    model.preprocessor = nn.Linear(int(np.prod(cifar.SHAPE)), int(np.prod(cifar.SHAPE)))
    flow = simulation_factory.build_spiking_hybrid_flow(_lif_pipeline(device), packed, model=model)
    assert {t.device.type for t in flow.parameters()} == {"cpu"}
    assert flow.preprocessor is model.preprocessor
    seen = _record_devices(monkeypatch)
    with torch.no_grad():
        flow(cifar._samples(2).reshape(2, -1))
    assert set(seen["flow"]) == {"cpu"} and set(seen["host"]) == {"cpu"}
