"""CH-C1 Readout Law: the step, the Soft Core Mapping hook, the atol-0 edges and the fixture's representability."""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import mimarsinan.pipelining.core.gates.value_gates as gates
import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as step_module
from mimarsinan.certification.value_certificate import VALUE_LAW_ATOL, VALUE_TWIN_FP64_ATOL, value_edge_atol
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import parse_deployment_document
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.mapping.export.law_solver import ReadoutLawDeclarationError
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.pipelines.deployment_specs import get_pipeline_step_specs
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
    ReadoutLawCalibrationError,
    ReadoutLawStep,
    clipping_census,
    install_readout_quantizers,
)
from unit.chip_simulation.value_run.integer_programs import LAW, build_integer_program, mlp

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "integer_value" / "mvm_int_readout_lenet5.json"


class _Pipeline:
    """The step's view of a pipeline: a config and an entry cache."""

    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _model():
    torch.manual_seed(0)
    first, second = Perceptron(6, 4), Perceptron(3, 6)
    return nn.Sequential(first, second), [first, second]


def _run_step(monkeypatch, platform, config=None, unreached=()):
    model, perceptrons = _model()
    perceptrons = perceptrons + list(unreached)
    model.get_perceptrons = lambda: perceptrons
    batch = torch.randn(64, 4)
    monkeypatch.setattr(step_module, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    monkeypatch.setattr(step_module, "iter_val_batches", lambda trainer, n: [(batch, None)])
    pipeline = _Pipeline(
        {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, **(config or {})},
        {"model": model, "platform_constraints_resolved": platform},
    )
    ReadoutLawStep(pipeline).run()
    return pipeline, perceptrons, batch


class TestPlacement:
    def test_applies_only_when_readout_law_is_declared(self):
        assert ReadoutLawStep.applies_to(DeploymentPlan.resolve({"core_semantics": "mvm", "readout_law": LAW}))
        assert not ReadoutLawStep.applies_to(DeploymentPlan.resolve({"core_semantics": "mvm"}))

    def test_the_step_follows_boundary_quantization_only_when_declared(self):
        document = json.loads(FIXTURE.read_text())
        params, platform = dict(document["deployment_parameters"]), dict(document["platform_constraints"])
        names = [name for name, _cls in get_pipeline_step_specs(merge_pipeline_config(params, platform))]
        assert names[names.index("Boundary Quantization") + 1] == "Readout Law"
        platform.pop("readout_law")
        undeclared = [name for name, _cls in get_pipeline_step_specs(merge_pipeline_config(params, platform))]
        assert undeclared == [name for name in names if name != "Readout Law"]

    def test_soft_core_mapping_calls_the_hook_after_ir_pruning(self):
        path = ROOT / "src" / "mimarsinan" / "pipelining" / "pipeline_steps" / "mapping" / "soft_core_mapping_step.py"
        process = next(n for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.FunctionDef) and n.name == "process")
        calls = [ast.unparse(c.func) for c in ast.walk(process) if isinstance(c, ast.Call)]
        lines = {name: next(c.lineno for c in ast.walk(process) if isinstance(c, ast.Call)
                            and ast.unparse(c.func) == name) for name in ("apply_ir_pruning_if_enabled",
                                                                          "install_readout_laws")}
        assert "install_readout_laws" in calls
        assert lines["apply_ir_pruning_if_enabled"] < lines["install_readout_laws"]


class TestStep:
    def test_installs_and_calibrates_a_quantizer_on_every_perceptron(self, monkeypatch):
        pipeline, perceptrons, batch = _run_step(monkeypatch, {"readout_law": LAW, "activation_bits": 8})
        assert all(isinstance(p.scaler, ReadoutQuantizer) for p in perceptrons)
        first_output = perceptrons[0].layer(batch).detach().abs()
        assert float(perceptrons[0].scaler.range) == pytest.approx(float(torch.quantile(first_output, 0.999)),
                                                                    rel=1e-5)
        report = pipeline.entries["readout_law_report"]
        assert set(report["perceptrons"]) == {p.name for p in perceptrons}
        assert all(0.0 <= row["clipped_fraction"] < 0.01 for row in report["perceptrons"].values())

    def test_a_fixed_range_is_the_declared_value(self, monkeypatch):
        law = {"adc_bits": 8, "range": "fixed", "range_value": 0.25, "mult_bits": 16}
        pipeline, perceptrons, _ = _run_step(monkeypatch, {"readout_law": law, "activation_bits": 8})
        assert all(float(p.scaler.range) == 0.25 for p in perceptrons)
        assert pipeline.entries["readout_law_report"]["range_fixed"] is True

    def test_without_activation_bits_refuses_by_name(self, monkeypatch):
        with pytest.raises(ReadoutLawDeclarationError, match="activation_bits"):
            _run_step(monkeypatch, {"readout_law": LAW})

    def test_without_weight_quantization_refuses_by_name(self, monkeypatch):
        with pytest.raises(ReadoutLawDeclarationError, match="weight quantization"):
            _run_step(monkeypatch, {"readout_law": LAW, "activation_bits": 8}, {"weight_quantization": False})

    def test_a_perceptron_the_calibration_never_reaches_refuses_by_name(self, monkeypatch):
        unreached = Perceptron(2, 6, name="unreached")
        with pytest.raises(ReadoutLawCalibrationError, match="never reached perceptron.*unreached"):
            _run_step(monkeypatch, {"readout_law": LAW, "activation_bits": 8}, unreached=[unreached])
        assert issubclass(ReadoutLawCalibrationError, ToolchainRefusal)

    def test_a_taken_scaler_slot_refuses(self):
        _model_, perceptrons = _model()
        perceptrons[0].set_scaler(nn.Tanh())
        with pytest.raises(ReadoutLawDeclarationError, match="scaler"):
            install_readout_quantizers(perceptrons, 8)

    def test_the_unstamped_quantizer_is_a_straight_through_estimator(self):
        quantizer = ReadoutQuantizer(4, range_value=7.0)
        y = torch.tensor([0.4, 2.5, -9.0], requires_grad=True)
        out = quantizer(y)
        assert out.tolist() == [0.0, 3.0, -7.0]
        out.sum().backward()
        assert y.grad.tolist() == [1.0, 1.0, 1.0]

    def test_clipping_census_counts_outputs_beyond_the_range(self):
        _model_, perceptrons = _model()
        quantizers = install_readout_quantizers(perceptrons[:1], 8)
        quantizers[0].calibrate(0.1)
        census = clipping_census(perceptrons[0], perceptrons[:1], quantizers, [torch.randn(32, 4)])
        row = census[perceptrons[0].name]
        assert row["outputs"] == 32 * 6 and row["clipped"] > 0


def _gate_pipeline():
    return SimpleNamespace(config={"core_semantics": "mvm", "device": "cpu"}, cache={})


class TestAtolZeroEdges:
    def test_the_edge_tolerance_is_zero_with_laws_only(self):
        assert value_edge_atol(True) == VALUE_LAW_ATOL == 0.0
        assert value_edge_atol(False) == VALUE_TWIN_FP64_ATOL

    def test_r_and_c_edges_certify_at_atol_zero(self, monkeypatch):
        program = build_integer_program(mlp(), (8,), num_classes=4)
        monkeypatch.setattr(gates, "_certificate_samples", lambda pipeline, model, n: torch.randn(n, 8) * 2)
        gates.run_model_value_parity_gate(_gate_pipeline(), program.model, program.ir)
        hybrid = program.packed_mapping(axons=256, neurons=5)
        certificate = gates.run_value_twin_certificate_gate(_gate_pipeline(), program.model, program.ir, hybrid)
        assert certificate.passed and certificate.atol == 0.0 and certificate.neuron_windows_compared > 0

    def test_a_one_code_defect_fails_the_c_edge_at_atol_zero(self, monkeypatch):
        program = build_integer_program(mlp(), (8,), num_classes=4)
        hybrid = program.packed_mapping(axons=256, neurons=5)
        hc = next(s for s in hybrid.stages if s.kind == "neural").hard_core_mapping.cores[0]
        hc.readout_law = hc.readout_law.splice(0, _one_code_up(hc.readout_law.take_columns([0])))
        monkeypatch.setattr(gates, "_certificate_samples", lambda pipeline, model, n: torch.randn(n, 8) * 2)
        with pytest.raises(RuntimeError, match="value twin certificate FAILED"):
            gates.run_value_twin_certificate_gate(_gate_pipeline(), program.model, program.ir, hybrid)

    def test_a_one_code_defect_fails_the_r_edge_at_atol_zero(self, monkeypatch):
        program = build_integer_program(mlp(), (8,), num_classes=4)
        core = next(n for n in program.ir.nodes if isinstance(n, NeuralCore))
        core.readout_law = _one_code_up(core.readout_law)
        monkeypatch.setattr(gates, "_certificate_samples", lambda pipeline, model, n: torch.randn(n, 8) * 2)
        with pytest.raises(RuntimeError, match="R-edge.*atol=0"):
            gates.run_model_value_parity_gate(_gate_pipeline(), program.model, program.ir)


def _one_code_up(law: CoreLaw) -> CoreLaw:
    """The same law with every output one code higher (B + 2^s)."""
    fields = {f.name: getattr(law, f.name) for f in dataclasses.fields(CoreLaw)}
    fields["bias"] = law.bias + np.left_shift(np.int64(1), law.shift)
    return CoreLaw(**fields)


class TestFixture:
    def test_the_fixture_round_trips_through_the_wizard_entry_points(self):
        config = json.loads(FIXTURE.read_text())
        assert parse_deployment_document(config).unknown == []
        assert json.dumps(emit_deployment_config(config), sort_keys=True) == json.dumps(config, sort_keys=True)
        resolution = resolve_draft(config)
        assert resolution.errors == [] and resolution.unknown_keys == []
        assert validate_deployment_config(config) == []

    def test_the_fixture_declares_the_integer_path_and_its_budgets(self):
        config = json.loads(FIXTURE.read_text())
        platform, params = config["platform_constraints"], config["deployment_parameters"]
        assert platform["activation_bits"] == 8 and platform["max_schedule_passes"] == 128
        assert set(platform["readout_law"]) >= {"adc_bits", "range", "mult_bits"}
        assert params["weight_quantization"] and params["pruning"] and params["allow_scheduling"]
        assert params["value_parity_samples"] == 128  # one validation batch per certificate edge
