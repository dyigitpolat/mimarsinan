"""T-2: the Model Building step's static gate measures the program the deployment converts (U-GATE).

The gate and the Torch Mapping step read ONE conversion object (the plan's packaging contract and its leaf
decomposition): under a value-domain configuration the gate passes and prints the deployed fraction; under a spiking
configuration of the same model it still refuses by name, where the refusal is the model's true placement.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config

from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.verification import onchip_fraction as of
from mimarsinan.mapping.verification.onchip_majority import OnchipMajorityError
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.config import torch_mapping_step as tms
from mimarsinan.pipelining.pipeline_steps.config.model_building_step import ModelBuildingStep
from mimarsinan.search.problems.joint.model_build import DeployedConversion

SHAPE = (3, 16, 16)
CLASSES = 10


class _InvertedResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU6())
        self.expand = nn.Sequential(nn.Conv2d(8, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU6())
        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.ReLU6())
        self.project = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, CLASSES)

    def forward(self, x):
        x = self.stem(x)
        x = x + self.project(self.depthwise(self.expand(x)))
        return self.head(torch.flatten(self.pool(x), 1))


def _net():
    torch.manual_seed(0)
    return _InvertedResidualNet().eval()


def _pipeline(*, mvm: bool, **extra) -> MockPipeline:
    config = default_config()
    config.update({"input_shape": SHAPE, "num_classes": CLASSES}, **extra)
    if mvm:
        config.update({"core_semantics": "mvm", "activation_quantization": True})
    return MockPipeline(config=config)


def _step(pipeline) -> ModelBuildingStep:
    step = ModelBuildingStep(pipeline)
    step.name = "ModelBuilding"
    return step


def _deployed(pipeline) -> DeployedConversion:
    return DeployedConversion(packaging=packaging_contract_for(DeploymentPlan.of(pipeline)),
                              decompose_leaf_modules=bool(pipeline.config.get("decompose_leaf_modules", False)))


_PRINTED = re.compile(
    r"static on-chip parameter majority: ([0-9.]+)% on chip \(on-chip=(\d+), host=(\d+), total=(\d+)\)")


def test_t2_value_domain_configuration_passes_and_prints_the_deployed_fraction(capsys):
    pipeline = _pipeline(mvm=True)
    _step(pipeline)._run_static_onchip_majority_gate(_net())  # the release refused here at 0.00%
    printed = _PRINTED.search(capsys.readouterr().out)
    assert printed is not None
    flow = tms.convert_pipeline_model(_net(), pipeline.config, _deployed(pipeline))
    ref = of.estimate_onchip_fraction(flow, SHAPE, CLASSES)
    assert (int(printed.group(2)), int(printed.group(3)), int(printed.group(4))) == (ref.onchip, ref.host, ref.total)
    assert ref.fraction > 0.5


def test_t2_spiking_configuration_of_the_same_model_still_refuses_by_name():
    pipeline = _pipeline(mvm=False)
    with pytest.raises(OnchipMajorityError, match=r"only 0\.00% of the \d+ params are estimated on chip"):
        _step(pipeline)._run_static_onchip_majority_gate(_net())


def test_t2_the_gate_and_the_torch_mapping_step_convert_under_one_object(monkeypatch):
    """The gate's conversion and the Torch Mapping step's receive the same packaging and leaf decomposition: the
    deployment's (here a value-domain plan that declares the leaf decomposition)."""
    from mimarsinan.torch_mapping import converter

    seen = []
    real = converter.convert_torch_model

    def spy(*args, **kwargs):
        seen.append((kwargs.get("packaging", "<default>"), kwargs.get("decompose_leaf_modules", "<default>")))
        return real(*args, **kwargs)

    monkeypatch.setattr(converter, "convert_torch_model", spy)
    monkeypatch.setattr(tms, "convert_torch_model", spy)
    pipeline = _pipeline(mvm=True, decompose_leaf_modules=True)
    expected = _deployed(pipeline)
    _step(pipeline)._run_static_onchip_majority_gate(_net())
    tms.convert_pipeline_model(_net(), pipeline.config, expected)
    assert seen == [(expected.packaging, True), (expected.packaging, True)], seen


def test_t2_the_pipelines_conversion_reads_the_object_not_the_config(monkeypatch):
    """``convert_pipeline_model`` takes packaging AND leaf decomposition from the conversion it is given."""
    seen = {}

    def spy(*args, **kwargs):
        seen.update(packaging=kwargs.get("packaging"), decompose=kwargs.get("decompose_leaf_modules"),
                    value_stage=kwargs.get("value_program_stage"))
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(tms, "convert_torch_model", spy)
    pipeline = _pipeline(mvm=True)
    conversion = DeployedConversion(packaging=packaging_contract_for(DeploymentPlan.of(pipeline)),
                                    decompose_leaf_modules=True)
    with pytest.raises(RuntimeError, match="stop after the call"):
        tms.convert_pipeline_model(_net(), pipeline.config, conversion)
    assert seen == {"packaging": conversion.packaging, "decompose": True, "value_stage": True}


def test_t2_the_pipelines_conversion_refuses_another_type():
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING

    pipeline = _pipeline(mvm=True)
    with pytest.raises(TypeError, match="DeployedConversion"):
        tms.convert_pipeline_model(_net(), pipeline.config, MVM_PACKAGING)


class _PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, stride=2), nn.BatchNorm2d(8), nn.ReLU())
        self.head = nn.Linear(8 * 8 * 8, CLASSES)

    def forward(self, x):
        return self.head(torch.flatten(self.c2(self.c1(x)), 1))


def test_t2_the_span_report_reads_the_deployed_conversion(capsys):
    """A streamed configuration's span report at build reads the deployment's conversion (spiking here)."""
    config = default_config()
    config.pop("spiking_mode")
    config.update({"input_shape": SHAPE, "num_classes": CLASSES, "spiking_family": "lif",
                   "spiking_variant": "streamed"})
    torch.manual_seed(0)
    _step(MockPipeline(config=config))._print_streamed_span_report(_PlainNet().eval())
    assert "[ModelBuildingStep] streamed span topology" in capsys.readouterr().out


def test_t2_the_torch_mapping_step_passes_the_deployed_conversion(monkeypatch):
    """The step's own call: the object it hands the conversion is the run's ``DeployedConversion``."""
    seen = {}

    class _Stop(Exception):
        pass

    def spy(native_model, config, conversion):
        seen["conversion"] = conversion
        raise _Stop

    monkeypatch.setattr(tms, "convert_pipeline_model", spy)
    pipeline = _pipeline(mvm=True, decompose_leaf_modules=True)
    pipeline.seed("model", _net())
    step = tms.TorchMappingStep(pipeline)
    step.name = "TorchMapping"
    pipeline.prepare_step(step)
    with pytest.raises(_Stop):
        step.process()
    assert type(seen["conversion"]).__name__ == "DeployedConversion", type(seen["conversion"]).__name__
    assert seen["conversion"] == _deployed(pipeline)
