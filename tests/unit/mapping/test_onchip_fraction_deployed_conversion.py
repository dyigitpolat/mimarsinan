"""The static on-chip estimators read the deployment's own conversion, never a default contract.

The defect (U-GATE): the Model Building gate converted a native model under the converter's default spiking
packaging contract while the deployment converts under the plan's (``TorchMappingStep``: a value-domain plan
packages any weight-stationary affine op and keeps its activations on the host). A model whose weight ops feed no
absorbable activation (an inverted residual: Conv-BN-ReLU6 expansion, a grouped Conv-BN-ReLU6, a Conv-BN projection
into a residual add, a bare Linear head) then read 0.00% on chip while its deployment places nearly all of it on chip.

T-1 value-domain parity, T-3 parity across contracts with the mapped IR, T-4 no silent default (a named refusal and
a source guard), T-5 no mutation; the search route's constraint and census on the same object.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import (
    MVM_PACKAGING,
    SPIKING_PACKAGING,
    packaging_contract_for,
)
from mimarsinan.mapping.verification import onchip_fraction as of
from mimarsinan.mapping.verification.onchip_majority import OnchipMajorityError, count_host_params
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
# The conversion object, importable from its first home (the search route) and its neutral home alike.
from mimarsinan.search.problems.joint.model_build import DeployedConversion

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
SHAPE = (3, 16, 16)
CLASSES = 10


class _InvertedResidualNet(nn.Module):
    """The failing form: no weight op feeds an absorbable activation (ReLU6 is not one, the projection and the head
    feed none), and one convolution is grouped (a host unit under either contract)."""

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


class _PlainNet(nn.Module):
    """A plain Conv-BN-ReLU stack with a Linear head (every weight op folds an activation but the head)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, stride=2), nn.BatchNorm2d(8), nn.ReLU())
        self.head = nn.Linear(8 * 8 * 8, CLASSES)

    def forward(self, x):
        return self.head(torch.flatten(self.c2(self.c1(x)), 1))


def _net(cls=_InvertedResidualNet):
    torch.manual_seed(0)
    return cls().eval()


def _config(*, mvm: bool, activation_quantization: bool = True, decompose: bool | None = None) -> dict:
    config = {
        "device": "cpu", "target_tq": 4, "weight_bits": 8, "spiking_mode": "lif", "firing_mode": "Default",
        "spike_generation_mode": "Deterministic", "thresholding_mode": "<", "simulation_steps": 4,
        "input_shape": SHAPE, "num_classes": CLASSES, "encoding_layer_placement": "subsume",
        "activation_quantization": bool(activation_quantization) if mvm else False,
    }
    if mvm:
        config["core_semantics"] = "mvm"
    if decompose is not None:
        config["decompose_leaf_modules"] = bool(decompose)
    return config


def _conversion(config: dict) -> DeployedConversion:
    """The deployment's conversion for ``config``, spelled from the plan's contract (the form of
    ``deployed_conversion_of``)."""
    plan = DeploymentPlan.resolve(config)
    return DeployedConversion(packaging=packaging_contract_for(plan),
                              decompose_leaf_modules=bool(config.get("decompose_leaf_modules", False)))


def _gate_step(config: dict):
    from mimarsinan.pipelining.pipeline_steps.config.model_building_step import ModelBuildingStep

    step = ModelBuildingStep.__new__(ModelBuildingStep)
    step.pipeline = SimpleNamespace(config=config)
    return step


def _parameter_bearing_host_units(flow):
    units, seen = [], set()
    for node in of._exec_nodes(flow):
        unit = of._host_unit(node)
        if unit is None or id(unit) in seen:
            continue
        seen.add(id(unit))
        if of._module_params(unit) > 0:
            units.append(unit)
    return units


def _grouped_conv_inside(module) -> bool:
    return any(isinstance(m, nn.Conv2d) and m.groups > 1 for m in module.modules())


# ---------------------------------------------------------------------------------------------------------- T-1
def test_t1_value_domain_gate_reads_the_deployed_program():
    """Under an MVM plan with activation quantization the static gate passes, and its parameter estimate equals the
    estimate on the pipeline's own conversion (``convert_pipeline_model`` under the plan's conversion): host, on-chip
    and total equal, the grouped convolution the only parameter-bearing host unit."""
    from mimarsinan.pipelining.pipeline_steps.config.torch_mapping_step import convert_pipeline_model

    config = _config(mvm=True)
    _gate_step(config)._run_static_onchip_majority_gate(_net())  # the release refused here: 0.00% on chip

    conversion = _conversion(config)
    assert conversion.packaging.is_value_domain and conversion.packaging.boundary_is_gridded
    est = of.assert_onchip_majority_estimate_or_raise(
        _net(), SHAPE, CLASSES, min_fraction=0.2, conversion=conversion)
    flow = convert_pipeline_model(_net(), config, conversion)
    ref = of.estimate_onchip_fraction(flow, SHAPE, CLASSES)
    assert (est.host, est.onchip, est.total) == (ref.host, ref.onchip, ref.total)
    assert est.onchip > 0 and est.fraction > 0.5
    host_units = _parameter_bearing_host_units(flow)
    assert len(host_units) == 1 and _grouped_conv_inside(host_units[0]), [type(u).__name__ for u in host_units]
    assert est.host == of._module_params(host_units[0])


# ---------------------------------------------------------------------------------------------------------- T-3
def _authoritative_breakdown(model, conversion):
    """The REAL decomposition under ``conversion``: convert, map to an IRGraph, count its host parameters."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.torch_mapping.converter import convert_torch_model

    flow = convert_torch_model(model, SHAPE, CLASSES, encoding_layer_placement="subsume",
                               packaging=conversion.packaging, decompose_leaf_modules=conversion.decompose_leaf_modules)
    total = int(sum(p.numel() for p in flow.parameters()))
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    ir_graph = IRMapping(q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=True).map(mapper_repr)
    return int(count_host_params(ir_graph)), total


_CONTRACTS = [
    pytest.param(dict(mvm=True, activation_quantization=False), id="mvm-float-boundary"),
    pytest.param(dict(mvm=True, activation_quantization=True), id="mvm-grid-boundary"),
    pytest.param(dict(mvm=True, activation_quantization=True, decompose=True), id="mvm-grid-decomposed"),
    pytest.param(dict(mvm=False), id="spiking"),
]
_MODELS = [
    pytest.param(_InvertedResidualNet, id="inverted-residual-grouped"),
    pytest.param(_PlainNet, id="plain-conv"),
]


@pytest.mark.parametrize("contract", _CONTRACTS)
@pytest.mark.parametrize("model_cls", _MODELS)
def test_t3_param_estimate_reproduces_authoritative_gate_under_the_plan(contract, model_cls):
    """The static estimate equals the mapped IR's breakdown under the SAME plan, value-domain contracts included
    (the release's parity cases were spiking only, the gap that let the defect pass)."""
    conversion = _conversion(_config(**contract))
    host, total = _authoritative_breakdown(_net(model_cls), conversion)
    est = of.estimate_onchip_fraction(_net(model_cls), SHAPE, CLASSES, metric="params", conversion=conversion)
    assert (est.host, est.total, est.onchip) == (host, total, total - host)


def test_t3_the_contracts_read_different_programs_on_the_failing_form():
    """The case the parity cases missed: the same model reads 0 on chip under the spiking contract and most of its
    parameters on chip under the value-domain one."""
    spiking = of.estimate_onchip_fraction(_net(), SHAPE, CLASSES, conversion=DeployedConversion())
    value = of.estimate_onchip_fraction(_net(), SHAPE, CLASSES, conversion=_conversion(_config(mvm=True)))
    assert spiking.onchip == 0 and value.fraction > 0.5


# ---------------------------------------------------------------------------------------------------------- T-4
_NATIVE_CALLS = [
    pytest.param(lambda m, **k: of.estimate_onchip_fraction(m, SHAPE, CLASSES, **k), id="estimate_onchip_fraction"),
    pytest.param(lambda m, **k: of.estimate_onchip_fractions(m, SHAPE, CLASSES, **k), id="estimate_onchip_fractions"),
    pytest.param(lambda m, **k: of.assert_onchip_majority_estimate_or_raise(m, SHAPE, CLASSES, min_fraction=0.0, **k),
                 id="assert_onchip_majority_estimate_or_raise"),
    pytest.param(lambda m, **k: of.classify_validity(m, SHAPE, CLASSES, **k), id="classify_validity"),
]


@pytest.mark.parametrize("call", _NATIVE_CALLS)
def test_t4_a_native_model_without_a_conversion_is_refused_by_name(call):
    with pytest.raises(ValueError) as refused:
        call(_net())
    assert type(refused.value).__name__ == "ConversionUnstatedError", repr(refused.value)
    assert "deployment's conversion" in str(refused.value)


def test_t4_the_streamed_span_report_refuses_the_same_way():
    from mimarsinan.mapping.verification.streamed import streamed_span_report_model

    with pytest.raises(ValueError) as refused:
        streamed_span_report_model(_net(), SHAPE, CLASSES)
    assert type(refused.value).__name__ == "ConversionUnstatedError"


@pytest.mark.parametrize("call", _NATIVE_CALLS)
def test_t4_a_conversion_of_another_type_is_refused(call):
    with pytest.raises(TypeError, match="DeployedConversion"):
        call(_net(), conversion=MVM_PACKAGING)


def test_t4_a_flow_is_measured_as_born_without_a_conversion():
    from mimarsinan.torch_mapping.converter import convert_torch_model

    flow = convert_torch_model(_net(), SHAPE, CLASSES, packaging=MVM_PACKAGING)
    as_born = of.estimate_onchip_fraction(flow, SHAPE, CLASSES)
    restated = of.estimate_onchip_fraction(flow, SHAPE, CLASSES, conversion=DeployedConversion(SPIKING_PACKAGING))
    assert (as_born.host, as_born.total) == (restated.host, restated.total)
    assert as_born.onchip > 0


def _mapped_ir(conversion):
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.torch_mapping.converter import convert_torch_model

    flow = convert_torch_model(_net(), SHAPE, CLASSES, packaging=conversion.packaging,
                               decompose_leaf_modules=conversion.decompose_leaf_modules)
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    return IRMapping(q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
                     allow_coalescing=False, hardware_bias=True).map(mapper_repr)


def test_t4_the_authoritative_gate_and_the_partition_read_a_native_model_on_its_conversion():
    """The post-mapping gate and the sealed partition measure their model's MACs through the same estimator: a
    native model is read on the conversion they are given, and refused by name without one."""
    from mimarsinan.deployment_record.build.partition import compute_partition_record

    conversion = _conversion(_config(mvm=True))
    ir_graph = _mapped_ir(conversion)
    expected = of.estimate_onchip_fraction(_net(), SHAPE, CLASSES, metric="macs", conversion=conversion)
    report = of.assert_onchip_validity_or_raise(ir_graph, _net(), SHAPE, CLASSES, floor=0.0, majority=0.0,
                                                conversion=conversion)
    assert (report.mac_estimate.host, report.mac_estimate.total) == (expected.host, expected.total)
    record = compute_partition_record(ir_graph, _net(), SHAPE, CLASSES, encoding_placement="subsume",
                                      conversion=conversion)
    assert (record.host_macs, record.total_macs) == (expected.host, expected.total)
    with pytest.raises(ValueError) as refused:
        of.assert_onchip_validity_or_raise(ir_graph, _net(), SHAPE, CLASSES, floor=0.0, majority=0.0)
    assert type(refused.value).__name__ == "ConversionUnstatedError"


#: The static estimators whose every call in ``src`` states its conversion (a flow is measured as born either way).
_GUARDED = ("_build_flow", "estimate_onchip_fraction", "estimate_onchip_fractions",
            "assert_onchip_majority_estimate_or_raise", "classify_validity", "streamed_span_report_model")


def test_t4_the_guarded_estimators_take_a_conversion():
    import inspect

    from mimarsinan.mapping.verification.streamed import streamability

    for name in _GUARDED:
        fn = getattr(of, name, None) or getattr(streamability, name)
        assert "conversion" in inspect.signature(fn).parameters, name


def test_t4_every_static_estimate_in_src_states_its_conversion():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _GUARDED and not any(k.arg == "conversion" for k in node.keywords):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {name}")
    assert not offenders, "static on-chip estimates without a stated conversion: " + "; ".join(offenders)


# ---------------------------------------------------------------------------------------------------------- T-5
def _snapshot(model):
    torch.manual_seed(1)
    x = torch.randn(2, *SHAPE)
    with torch.no_grad():
        out = model(x).clone()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    kinds = [(name, type(m).__name__) for name, m in model.named_modules()]
    return state, kinds, x, out


def _assert_unchanged(model, before):
    state, kinds, x, out = before
    after = model.state_dict()
    assert list(after) == list(state)
    assert all(torch.equal(after[k], state[k]) for k in state)
    assert [(name, type(m).__name__) for name, m in model.named_modules()] == kinds
    with torch.no_grad():
        assert torch.equal(model(x), out)


@pytest.mark.parametrize("mvm", [True, False], ids=["mvm", "spiking"])
def test_t5_the_gate_leaves_the_native_model_unchanged(mvm):
    model = _net()
    before = _snapshot(model)
    try:
        _gate_step(_config(mvm=mvm))._run_static_onchip_majority_gate(model)
    except OnchipMajorityError:
        pass  # the spiking reading of this form refuses; the model must still be untouched
    _assert_unchanged(model, before)


@pytest.mark.parametrize("mvm", [True, False], ids=["mvm", "spiking"])
def test_t5_the_estimator_leaves_the_native_model_unchanged(mvm):
    model = _net()
    before = _snapshot(model)
    of.estimate_onchip_fractions(model, SHAPE, CLASSES, metrics=("params", "macs", "ops"),
                                 conversion=_conversion(_config(mvm=mvm)))
    _assert_unchanged(model, before)


# ---------------------------------------------------------------------------------------- the search route (F-3)
def test_the_search_constraint_reads_the_problems_conversion():
    """``JointConstrainMixin.onchip_fraction`` measures a raw candidate under the problem's conversion."""
    from mimarsinan.search.problems.joint.constrain import JointConstrainMixin

    conversion = _conversion(_config(mvm=True))
    problem = SimpleNamespace(
        input_shape=SHAPE, num_classes=CLASSES, conversion=conversion,
        candidate_encoding_placement=lambda configuration: "subsume",
        _candidate_model=lambda mc, pcfg, placement: (_net(), 0.0),
    )
    fraction = JointConstrainMixin.onchip_fraction(problem, {"model_config": {}, "platform_constraints": {}})
    expected = of.estimate_onchip_fraction(_net(), SHAPE, CLASSES, conversion=conversion).fraction
    assert fraction == expected and fraction > 0.5


def test_the_search_census_reads_the_conversion_it_is_given():
    from mimarsinan.search.problems.joint.candidate_fragments import compute_onchip_census

    conversion = _conversion(_config(mvm=True))
    params, macs, ops = compute_onchip_census(_net(), SHAPE, CLASSES, "subsume", conversion=conversion)
    ref = of.estimate_onchip_fractions(_net(), SHAPE, CLASSES, metrics=("params", "macs", "ops"),
                                       conversion=conversion)
    assert [(e.host, e.total) for e in (params, macs, ops)] == [(e.host, e.total) for e in ref]
    assert params.onchip > 0
    with pytest.raises(ValueError) as refused:
        compute_onchip_census(_net(), SHAPE, CLASSES, "subsume")
    assert type(refused.value).__name__ == "ConversionUnstatedError"


def test_the_search_layout_census_reads_the_problems_conversion():
    """``JointLayoutMixin._onchip_census`` (the candidate's quantity census) hands the problem's conversion on."""
    from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin

    conversion = _conversion(_config(mvm=True))
    problem = SimpleNamespace(input_shape=SHAPE, num_classes=CLASSES, conversion=conversion, _searches_model=True,
                              _requires_fragment=lambda fragment: True, _hw_only_cache={})
    census = JointLayoutMixin._onchip_census(problem, _net(), "subsume")
    ref = of.estimate_onchip_fractions(_net(), SHAPE, CLASSES, metrics=("params", "macs", "ops"),
                                       conversion=conversion)
    assert [(e.host, e.total) for e in census] == [(e.host, e.total) for e in ref]
    assert census[0].onchip > 0


def test_the_conversion_object_has_one_home():
    """The search route's name and the neutral one are the same class (the verification layer never imports the
    search package for it)."""
    from mimarsinan.mapping.platform import packaging_contract

    assert packaging_contract.DeployedConversion is DeployedConversion
    assert copy.deepcopy(DeployedConversion()) == DeployedConversion(SPIKING_PACKAGING, False)
    for decompose in (True, False, None):
        config = _config(mvm=True, decompose=decompose)
        resolved = packaging_contract.deployed_conversion_of(DeploymentPlan.resolve(config), config)
        assert resolved == _conversion(config)
        assert resolved.decompose_leaf_modules is bool(decompose)
    spiking = _config(mvm=False)
    assert packaging_contract.deployed_conversion_of(DeploymentPlan.resolve(spiking), spiking) == DeployedConversion()
