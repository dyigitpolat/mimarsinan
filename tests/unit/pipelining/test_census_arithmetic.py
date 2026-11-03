"""[U-CEN] Every census of the deployed program counts what the deployed program computes.

A census is the toolchain's own read of the deployed program over the eval set: Soft and Hard Core Mapping's metric
flows (the identity and the packed value program, and their spiking twins). Its count is printed as the step metric
and compared with the deployed read (the harvest's K9: the packed value program at float64 on the CPU, where the
int64 golden reads the integer program exactly, K6; a spiking program's deployed read is its spiking executor on the
CPU). These tests pin every census to that arithmetic:

- a value program built so that float32 drops a 2^-46 term that float64 keeps exactly, which flips one sample's class
  with no tie (by construction, not found by search), is counted by both value censuses at its exact count;
- a census asked for on a CUDA pipeline runs on the CPU;
- each census prints its device and dtype in the step's log, and the static check declares the census's dtype.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, TinyDataProvider, default_config, make_tiny_ir_graph

import mimarsinan.pipelining.core.gates.value_gates as gates
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core import simulation_factory
from mimarsinan.torch_mapping.converter import convert_torch_model

_E = 2.0 ** -23
#: The three cores' matrices (input rows, then the always-on bias row), every entry exact in float32. Core 0 squares
#: 1 + 2^-23 (exact: 1 + 2^-22 + 2^-46; float32 rounds the 2^-46 away), core 1 subtracts 1 + 2^-22 (float32: 0,
#: float64: 2^-46), the host ReLU keeps the sign, core 2 lifts it by 2^44 above the 0.1 of class 0.
_CORES = ([[1 + _E], [0.0]], [[1.0], [-(1 + 2 * _E)]], [[0.0, 2.0 ** 44], [0.1, 0.0]])
_INPUTS = (1 + _E, 0.5, 2.0)
_LABELS = (1, 0, 1)


class _Cache(dict):
    def add(self, key, obj, strategy="basic"):
        self[key] = obj


class _Provider(TinyDataProvider):
    def __init__(self, inputs, labels, input_shape, num_classes):
        super().__init__(input_shape=input_shape, num_classes=num_classes, size=len(labels))
        self._ds.data = torch.as_tensor(inputs, dtype=torch.float32).reshape(len(labels), *input_shape)
        self._ds.targets = torch.as_tensor(labels)


class _Factory:
    def __init__(self, *args):
        self._provider = _Provider(*args)

    def create(self):
        return self._provider


def _pipeline(config, inputs, labels):
    pipeline = MockPipeline(config=config, data_provider_factory=_Factory(
        inputs, labels, tuple(config["input_shape"]), int(config["num_classes"])))
    pipeline.cache = _Cache()
    return pipeline


def _value_pipeline(device="cpu"):
    config = default_config()
    config.update(core_semantics="mvm", device=device, input_shape=(1,), num_classes=2)
    return _pipeline(config, [[v] for v in _INPUTS], list(_LABELS))


def _value_program():
    """(IR graph, packed program) of the three-core value program with host ReLUs between its cores."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 2)).eval()
    flow = convert_torch_model(model, (1,), 2, device="cpu", packaging=MVM_PACKAGING).eval()
    mapper = flow.get_mapper_repr()
    mapper.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64).map(mapper)
    cores = [node for node in ir.nodes if isinstance(node, NeuralCore)]
    assert len(cores) == 3 and all(float(core.threshold) == 1.0 for core in cores)
    for core, rows in zip(cores, _CORES):
        core.core_matrix = np.asarray(rows, dtype=core.core_matrix.dtype)
    packed = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 9}])
    return ir, packed


def _exact_count(ir) -> int:
    """The program's count in exact rational arithmetic: the deployed program's function with no rounding."""
    matrices = [[[Fraction(float(v)) for v in row] for row in np.asarray(node.core_matrix)]
                for node in ir.nodes if isinstance(node, NeuralCore)]
    correct = 0
    for x, label in zip(_INPUTS, _LABELS):
        h = [Fraction(x)]
        for i, m in enumerate(matrices):
            out = [sum(h[k] * m[k][j] for k in range(len(h))) + m[len(h)][j] for j in range(len(m[0]))]
            h = [max(v, Fraction(0)) for v in out] if i < len(matrices) - 1 else out
        assert len(set(h)) == len(h), "the fixture holds no output tie"
        correct += int(max(range(len(h)), key=h.__getitem__) == label)
    return correct


def _count(flow, inputs, labels) -> int:
    with torch.no_grad():
        out = flow(torch.as_tensor(inputs, dtype=torch.float32).reshape(len(labels), -1))
    return int((out.argmax(1).cpu() == torch.as_tensor(labels)).sum())


def test_the_fixture_flips_one_class_in_float32_and_none_in_float64():
    """The construction itself: float32 loses one sample, float64 reads the exact count (no search)."""
    ir, packed = _value_program()
    exact = _exact_count(ir)
    assert exact == 3
    inputs = [[v] for v in _INPUTS]
    assert _count(ValueHybridCoreFlow(packed, dtype=torch.float32), inputs, _LABELS) == exact - 1
    assert _count(ValueHybridCoreFlow(packed, device="cpu", dtype=torch.float64), inputs, _LABELS) == exact


def test_the_hard_core_mapping_value_census_reads_the_exact_count():
    ir, packed = _value_program()
    acc = gates.run_value_mapping_metric(_value_pipeline(), ir, None, hybrid_mapping=packed)
    assert acc * len(_LABELS) == pytest.approx(_exact_count(ir))


def test_the_soft_core_mapping_value_census_reads_the_exact_count():
    ir, _packed = _value_program()
    acc = gates.run_value_identity_metric(_value_pipeline(), ir)
    assert acc * len(_LABELS) == pytest.approx(_exact_count(ir))


def test_the_value_census_equals_the_deployed_read():
    """The deployed read's executor, as the harvest builds it: the packed value program at float64 on the CPU."""
    ir, packed = _value_program()
    deployed = _count(ValueHybridCoreFlow(packed, device="cpu", dtype=torch.float64), [[v] for v in _INPUTS], _LABELS)
    acc = gates.run_value_mapping_metric(_value_pipeline(), ir, None, hybrid_mapping=packed)
    assert round(acc * len(_LABELS)) == deployed


def _record_value_runs(monkeypatch) -> list:
    runs = []
    execute = ValueHybridCoreFlow._execute

    def recording(self, x, *, retain_state):
        runs.append((str(self.execution_device), self.value_dtype))
        return execute(self, x, retain_state=retain_state)

    monkeypatch.setattr(ValueHybridCoreFlow, "_execute", recording)
    return runs


def test_a_value_census_asked_for_on_a_cuda_pipeline_runs_on_the_cpu(monkeypatch):
    ir, packed = _value_program()
    runs = _record_value_runs(monkeypatch)
    acc = gates.run_value_mapping_metric(_value_pipeline("cuda"), ir, None, hybrid_mapping=packed)
    assert acc * len(_LABELS) == pytest.approx(_exact_count(ir))
    acc = gates.run_value_identity_metric(_value_pipeline("cuda:0"), ir)
    assert acc * len(_LABELS) == pytest.approx(_exact_count(ir))
    assert runs and set(runs) == {("cpu", torch.float64)}


def test_every_value_census_prints_its_device_and_dtype(capsys):
    ir, packed = _value_program()
    gates.run_value_mapping_metric(_value_pipeline("cuda"), ir, None, hybrid_mapping=packed)
    gates.run_value_identity_metric(_value_pipeline(), ir)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[Census]")]
    assert len(lines) == 2
    assert all(re.search(r"device=cpu dtype=float64", ln) for ln in lines)
    assert "pipeline device cuda" in lines[0]


def _lif_program():
    np.random.seed(3)
    return make_tiny_ir_graph(in_dim=64, hidden_dim=4, out_dim=4)


def _lif_pipeline(device):
    config = default_config()
    config["device"] = device
    generator = torch.Generator().manual_seed(7)
    inputs = torch.rand(10, 1, 8, 8, generator=generator)
    labels = torch.randint(0, 4, (10,), generator=generator)
    return _pipeline(config, inputs, labels)


_LIF_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}]}


def _record_spiking_runs(monkeypatch) -> list:
    runs = []
    forward = SpikingHybridCoreFlow.forward

    def recording(self, x):
        runs.append(str(x.device))
        return forward(self, x)

    monkeypatch.setattr(SpikingHybridCoreFlow, "forward", recording)
    return runs


@pytest.mark.parametrize("census", ["packed", "identity"])
def test_a_spiking_census_asked_for_on_a_cuda_pipeline_runs_on_the_cpu(monkeypatch, census):
    run = (simulation_factory.run_hcm_mapping_metric if census == "packed"
           else simulation_factory.run_scm_identity_metric)
    on_cpu = run(_lif_pipeline("cpu"), _lif_program(), _LIF_CORES)
    runs = _record_spiking_runs(monkeypatch)
    asked_cuda = run(_lif_pipeline("cuda"), _lif_program(), _LIF_CORES)
    assert asked_cuda == on_cpu
    assert runs and set(runs) == {"cpu"}


def test_every_spiking_census_prints_its_device_and_dtype(capsys):
    simulation_factory.run_hcm_mapping_metric(_lif_pipeline("cuda"), _lif_program(), _LIF_CORES)
    simulation_factory.run_scm_identity_metric(_lif_pipeline("cpu"), _lif_program(), _LIF_CORES)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[Census]")]
    assert len(lines) == 2
    assert all("device=cpu" in ln and "cores float64" in ln for ln in lines)


def test_the_census_device_rule():
    from mimarsinan.pipelining.core.census.arithmetic import census_device

    assert census_device("cpu") == "cpu"
    assert census_device(torch.device("cpu")) == "cpu"
    assert census_device("cuda") == "cpu"
    assert census_device("cuda:1") == "cpu"
    assert census_device(None) == "cpu"


def test_the_static_check_declares_the_census_dtype():
    """Theorem 1's H8 reads the value census's upload dtype from the declaration: it is the census's own."""
    from mimarsinan.certification.program_check.declaration import VALUE_ACCUMULATION, VALUE_UPLOAD
    from mimarsinan.pipelining.core.census.arithmetic import VALUE_CENSUS_DTYPE

    name = str(VALUE_CENSUS_DTYPE).removeprefix("torch.")
    assert VALUE_UPLOAD == {"value_census": name}
    assert VALUE_ACCUMULATION["value_census"] == name
