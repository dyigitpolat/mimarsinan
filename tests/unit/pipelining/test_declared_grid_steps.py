"""The pipeline sites of the Wave 2 grid declarations: Boundary Quantization and Readout Law read the declared
quantizer state, Weight Quantization the declared scale and bias rules, both verification steps accept them."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step as bq_module
import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as rl_module
import mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step as wq_module
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.grid_declarations import (
    GridDeclarationError,
    ParameterGrids,
    projection_grids,
)
from mimarsinan.mapping.support.quantizer_state import QuantizerStateError, perceptron_fx_names
from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid, DeclaredBoundaryGrid
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.mapping.core_quantization_verification_step import (
    CoreQuantizationVerificationStep,
)
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import BoundaryQuantizationStep
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import ReadoutLawStep
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from unit.mapping.declared_programs import write_state


class _Pipeline:
    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)
        self.reporter = SimpleNamespace(report=lambda *a, **k: None)

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _flow():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4)).eval()
    return convert_torch_model(model, (8,), 4, device="cpu", packaging=MVM_PACKAGING).eval()


def _batches(monkeypatch, module):
    batch = torch.randn(32, 8, generator=torch.Generator().manual_seed(1))
    monkeypatch.setattr(module, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    monkeypatch.setattr(module, "iter_val_batches", lambda trainer, n: [(batch, None)])


def _boundary_step(monkeypatch, flow, **config):
    _batches(monkeypatch, bq_module)
    pipeline = _Pipeline({"activation_bits": 5, "device": "cpu", "core_semantics": "mvm", **config},
                         {"model": flow})
    BoundaryQuantizationStep(pipeline).run()
    return pipeline


def _readout_step(monkeypatch, flow, platform):
    _batches(monkeypatch, rl_module)
    pipeline = _Pipeline({"core_semantics": "mvm", "device": "cpu", "weight_quantization": True},
                         {"model": flow, "platform_constraints_resolved": platform})
    ReadoutLawStep(pipeline).run()
    return pipeline


LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}


class TestBoundaryQuantization:
    def test_the_declared_entry_grid_replaces_the_calibrated_one(self, monkeypatch, tmp_path):
        flow = _flow()
        declaration = write_state(tmp_path / "s.json", {
            "_0": {"entry": {"scale": 6.0, "offset": 3.0, "rounding": "floor", "codes": [0, 15]}}})
        _boundary_step(monkeypatch, flow, quantizer_state=declaration)
        names = perceptron_fx_names(flow.get_mapper_repr())
        grid = names["_0"].input_activation.grid
        assert isinstance(grid, DeclaredBoundaryGrid) and (grid.offset, grid.rounding) == (3.0, "floor")
        assert type(names["_2"].input_activation.grid) is BoundaryGrid

    def test_without_the_declaration_every_grid_is_calibrated(self, monkeypatch):
        flow = _flow()
        _boundary_step(monkeypatch, flow)
        for perceptron in flow.get_perceptrons():
            assert type(perceptron.input_activation.grid) is BoundaryGrid


class TestReadoutLaw:
    def test_declared_ranges_and_the_identity_output_law(self, monkeypatch, tmp_path):
        flow = _flow()
        declaration = write_state(tmp_path / "s.json", {"_0": {"readout_range": 2.5}})
        pipeline = _readout_step(monkeypatch, flow, {"readout_law": {**LAW, "output_law": "identity"},
                                                     "activation_bits": 5, "quantizer_state": declaration})
        names = perceptron_fx_names(flow.get_mapper_repr())
        assert float(names["_0"].scaler.range) == 2.5
        assert float(names["_2"].scaler.range) == 0.0  # inert: an identity output law requantizes nothing
        report = pipeline.entries["readout_law_report"]
        assert report["declared_ranges"] == 1 and report["identity_outputs"] == 1

    def test_the_inert_output_quantizer_is_the_identity_before_the_stamp(self, monkeypatch):
        flow = _flow()
        _readout_step(monkeypatch, flow, {"readout_law": {**LAW, "output_law": "identity"}, "activation_bits": 5})
        names = perceptron_fx_names(flow.get_mapper_repr())
        y = torch.randn(7, 4, dtype=torch.float32)
        assert torch.equal(names["_2"].scaler(y), y)

    def test_a_declared_range_on_an_identity_output_refuses_by_name(self, monkeypatch, tmp_path):
        flow = _flow()
        declaration = write_state(tmp_path / "s.json", {"_2": {"readout_range": 2.5}})
        with pytest.raises(QuantizerStateError, match="'_2'"):
            _readout_step(monkeypatch, flow, {"readout_law": {**LAW, "output_law": "identity"},
                                              "activation_bits": 5, "quantizer_state": declaration})

    def test_without_declarations_the_step_is_unchanged(self, monkeypatch):
        flow = _flow()
        pipeline = _readout_step(monkeypatch, flow, {"readout_law": LAW, "activation_bits": 5})
        assert all(float(p.scaler.range) > 0 for p in flow.get_perceptrons())
        assert "declared_ranges" not in pipeline.entries["readout_law_report"]


class TestWeightQuantizationDeclarations:
    def test_the_projection_grids_and_their_refusals(self):
        law, fractional = {"bias_resolution": "law"}, {"weight_scale_rule": "fractional"}
        def unread():
            raise AssertionError("the bias mode is read only under a declared law bias")

        assert projection_grids(fractional, two_scale=False, on_chip_bias=unread) == ParameterGrids(True, False)
        assert projection_grids({}, two_scale=True, on_chip_bias=unread) == ParameterGrids(False, False)
        assert projection_grids(law, two_scale=False, on_chip_bias=lambda: True) == ParameterGrids(False, True)
        with pytest.raises(GridDeclarationError, match="wq_two_scale_projection") as info:
            projection_grids(law, two_scale=True, on_chip_bias=lambda: True)
        assert isinstance(info.value, ToolchainRefusal)
        with pytest.raises(GridDeclarationError, match="has_bias"):
            projection_grids(law, two_scale=False, on_chip_bias=lambda: False)

    def test_the_step_hands_the_declared_grids_to_the_tuner(self, monkeypatch):
        flow = _flow()
        captured = {}
        monkeypatch.setattr(WeightQuantizationStep, "run_tuner",
                            lambda self, cls, *args, **kwargs: captured.update(kwargs))
        monkeypatch.setattr(WeightQuantizationStep, "_canonicalize_starved_bias_outliers",
                            lambda self, model: captured.setdefault("canonicalized", True))
        config = {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, "weight_bits": 4,
                  "weight_scale_rule": "fractional", "bias_resolution": "law", "has_bias": True}
        pipeline = _Pipeline(config, {"model": flow, "adaptation_manager": object()})
        WeightQuantizationStep(pipeline).process()
        assert captured["fractional_scale"] is True and captured["float_bias"] is True
        assert "canonicalized" not in captured  # a float bias starves no shared grid

    def test_the_tuner_projects_with_the_declared_grids(self):
        from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
            NormalizationAwarePerceptronQuantizationTuner as Tuner,
        )
        tuner = Tuner.__new__(Tuner)
        tuner.quantization_bits, tuner.two_scale_projection = 4, False
        tuner.fractional_scale, tuner.float_bias = True, True
        tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
        perceptron = Perceptron(3, 5)
        bias = perceptron.layer.bias.detach().clone()
        tuner._get_new_perceptron_transform(1.0)(perceptron)
        peak = float(perceptron.layer.weight.detach().abs().max())
        assert float(perceptron.parameter_scale) == pytest.approx(7.0 / peak, rel=1e-5)
        assert torch.equal(perceptron.layer.bias.detach(), bias)


class TestVerification:
    def test_a_float_bias_passes_the_wq_contract_only_when_declared(self):
        perceptron = Perceptron(3, 5)
        with torch.no_grad():
            perceptron.layer.bias.copy_(torch.tensor([0.123, -0.777, 0.5]))
        NormalizationAwarePerceptronQuantization(4, "cpu", 1.0, fractional_scale=True,
                                                 float_bias=True).transform(perceptron)
        assert_effective_parameters_on_chip_grid(perceptron, 7, law_bias=True)
        with pytest.raises(AssertionError):
            assert_effective_parameters_on_chip_grid(perceptron, 7)

    def test_core_quantization_verification_reads_the_declared_grids(self):
        core = NeuralCore(id=0, name="c", input_sources=np.array([IRSource(-2, 0), IRSource(-2, 1)]),
                          core_matrix=np.array([[3, -2], [7, 1]], dtype=np.int8), threshold=6.3951,
                          hardware_bias=np.array([1.37, -0.2]))
        core.parameter_scale = torch.tensor(1.0)
        ir = IRGraph(nodes=[core], output_sources=np.array([IRSource(0, 0), IRSource(0, 1)]))
        config = {"core_semantics": "mvm", "weight_quantization": True, "weight_bits": 4}
        declared = {**config, "weight_scale_rule": "fractional", "bias_resolution": "law"}
        CoreQuantizationVerificationStep(_Pipeline(declared, {"ir_graph": ir})).process()
        with pytest.raises(AssertionError):
            CoreQuantizationVerificationStep(_Pipeline(config, {"ir_graph": ir})).process()
