"""WI-5g: core types with their own weight bits and the layer-to-type rule.

The declaration (refusals by name), each layer's grid at its type's width, packing bound to the type, and the atol-0
certificate NF = SCM = HCM = int64 golden of mixed-width programs with lanes and without, single and multi-pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.build import effective_class, passthrough_keys
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.value_schema import schema_of, value_errors
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.type_binding import CoreTypeBindingError, binding_classes
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.core_types import (
    LAYER_RULE_KEY,
    CoreTypeBinding,
    check_core_codes,
    install_layer_weight_bits,
    refuse_unbuilt_domain,
    typed_cores_config,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS, LAYER_WEIGHT_BITS
from unit.chip_simulation.value_run.integer_programs import assert_exact, lenet5, mlp, residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, cores, lane_platform, real_inputs

CORE = {"max_axons": 256, "max_neurons": 32, "count": 999}
#: The b4442 pattern on the residual net: the last stage and the classifier on 2-bit cores, the rest on 4-bit ones.
LAST_STAGE = [{"layers": ["blocks_1_*", "fc"], "weight_bits": 2}]
GRIDS = {"weight_scale_rule": "fractional", "bias_resolution": "law"}


def typed(rule=None, widths=(4, 2), default=4, **extra) -> dict:
    platform = {"weight_bits": default, "cores": [dict(CORE, weight_bits=w) for w in widths], **extra}
    if rule is not None:
        platform[LAYER_RULE_KEY] = rule
    return platform


def _flow(model, shape, classes):
    flow = convert_torch_model(model.eval(), shape, classes, device="cpu", packaging=MVM_PACKAGING).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    return flow


class TestDeclaration:
    def test_nothing_binds_without_a_type_width_or_a_rule(self):
        platform = {"weight_bits": 4, "cores": [CORE]}
        assert CoreTypeBinding.from_config(platform) is None
        assert typed_cores_config(platform) is platform["cores"], "the declared list itself, untouched"
        assert install_layer_weight_bits(_flow(mlp(), (8,), 4), platform) is None

    def test_a_type_takes_the_platform_width_unless_it_declares_its_own(self):
        binding = CoreTypeBinding.from_config({"weight_bits": 4, "cores": [CORE, dict(CORE, weight_bits=2)]})
        assert binding is not None and binding.type_bits == (4, 2) and binding.rules == ()
        assert [c["weight_bits"] for c in binding.typed_cores([CORE, dict(CORE, weight_bits=2)])] == [4, 2]

    def test_the_rule_selects_layers_by_fx_glob_and_the_rest_take_the_platform_width(self):
        binding = CoreTypeBinding.from_config(typed(LAST_STAGE))
        assert binding is not None
        widths = binding.layer_bits(["stem", "blocks_0_conv1", "blocks_1_conv2", "blocks_1_shortcut_0", "fc"])
        assert widths == {"stem": 4, "blocks_0_conv1": 4, "blocks_1_conv2": 2, "blocks_1_shortcut_0": 2, "fc": 2}

    @pytest.mark.parametrize("platform,match", [
        (typed([{"layers": ["fc"], "weight_bits": 3}]), "no core type declares"),
        (typed([{"layers": "fc", "weight_bits": 2}]), "non-empty list"),
        (typed([{"layers": ["fc"]}]), "exactly"),
        (typed([{"layers": ["fc"], "weight_bits": 1}]), "at least 2"),
        ({"cores": [dict(CORE, weight_bits=2)]}, "platform weight_bits"),
        (typed({"layers": ["fc"], "weight_bits": 2}), "list of rules"),
    ])
    def test_a_malformed_or_unplaceable_declaration_is_refused_by_name(self, platform, match):
        with pytest.raises(CoreTypeBindingError, match=match):
            CoreTypeBinding.from_config(platform)

    def test_a_stale_glob_two_widths_for_one_layer_and_an_unplaced_default_are_refused(self):
        stale = CoreTypeBinding.from_config(typed([{"layers": ["conv9"], "weight_bits": 2}]))
        with pytest.raises(CoreTypeBindingError, match="matches no layer"):
            stale.layer_bits(["fc"])  # pyright: ignore[reportOptionalMemberAccess]
        both = CoreTypeBinding.from_config(typed([{"layers": ["f*"], "weight_bits": 2},
                                                  {"layers": ["fc"], "weight_bits": 4}]))
        with pytest.raises(CoreTypeBindingError, match="rules of weight widths"):
            both.layer_bits(["fc"])  # pyright: ignore[reportOptionalMemberAccess]
        only_two = CoreTypeBinding.from_config(typed([{"layers": ["fc"], "weight_bits": 2}], widths=(2,)))
        with pytest.raises(CoreTypeBindingError, match="no core type of that width"):
            only_two.layer_bits(["stem", "fc"])  # pyright: ignore[reportOptionalMemberAccess]

    def test_typed_widths_are_refused_on_spiking_cores_and_float_weights(self):
        with pytest.raises(CoreTypeBindingError, match="spiking"):
            refuse_unbuilt_domain(typed(LAST_STAGE), value_domain=False, weight_quantization=True)
        with pytest.raises(CoreTypeBindingError, match="float weights"):
            refuse_unbuilt_domain(typed(LAST_STAGE), value_domain=True, weight_quantization=False)
        refuse_unbuilt_domain({"weight_bits": 4, "cores": [CORE]}, value_domain=False, weight_quantization=False)


class TestLayerGrids:
    def test_each_layer_is_stamped_with_its_type_width_and_a_stale_stamp_is_refused(self):
        flow = _flow(residual_block_net(), (3, 16, 16), 10)
        widths = install_layer_weight_bits(flow, typed(LAST_STAGE))
        assert widths is not None and sorted(set(widths.values())) == [2, 4]
        assert [getattr(p, LAYER_WEIGHT_BITS) for p in flow.get_perceptrons()] == [4, 4, 4, 2, 2, 2, 2]
        with pytest.raises(CoreTypeBindingError, match="stale Weight Quantization"):
            install_layer_weight_bits(flow, typed([{"layers": ["fc"], "weight_bits": 2}]))

    def test_the_projection_quantizes_each_layer_on_its_type_grid(self):
        flow = _flow(residual_block_net(), (3, 16, 16), 10)
        install_layer_weight_bits(flow, typed(LAST_STAGE))
        napq = NormalizationAwarePerceptronQuantization(4, "cpu", rate=1.0, fractional_scale=True, float_bias=True)
        transformer = PerceptronTransformer()
        for perceptron in flow.get_perceptrons():
            napq.transform(perceptron)
            codes = torch.round(transformer.get_effective_weight(perceptron) * perceptron.parameter_scale)
            levels = set(codes.flatten().tolist())
            if getattr(perceptron, LAYER_WEIGHT_BITS) == 2:
                assert levels <= {-1.0, 0.0, 1.0} and {-1.0, 1.0} & levels, "3 levels: scale 1 / max|w|"
            else:
                assert levels <= set(map(float, range(-7, 8))) and len(levels) > 3


class TestTypeBoundPacking:
    def test_binding_classes(self):
        class Core:
            def __init__(self, bits=None):
                if bits is not None:
                    setattr(self, CORE_WEIGHT_BITS, bits)
        assert binding_classes([Core(), Core()], [Core()]) is None
        assert binding_classes([Core(2), Core(4), Core(2)], [Core(4), Core(2)]) == [4, 2]
        with pytest.raises(CoreTypeBindingError, match="carry no weight width"):
            binding_classes([Core(2), Core()], [Core(2)])
        with pytest.raises(CoreTypeBindingError, match="carry no weight width"):
            binding_classes([Core(2)], [Core(2), Core()])

    def test_every_core_lands_on_a_hard_core_of_its_layer_width(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                     platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True)
        width_of = {int(c.id): int(getattr(c, CORE_WEIGHT_BITS)) for c in cores(program)}
        assert sorted(set(width_of.values())) == [2, 4]
        mapping = program.packed_mapping(axons=256, neurons=32, count=999, widths=(4, 2))
        placed = 0
        for stage in mapping.stages:
            if stage.kind != "neural":
                continue
            segment = stage.hard_core_mapping
            for hard, placements in zip(segment.cores, segment.soft_core_placements_per_hard_core):
                for placement in placements:
                    assert getattr(hard, CORE_WEIGHT_BITS) == width_of[int(placement["ir_node_id"])]
                    placed += 1
        assert placed >= len(width_of)

    def test_a_spent_width_refuses_at_packing_like_any_capacity(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                     platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True)
        with pytest.raises(RuntimeError, match="No more hard cores"):
            program.packed_mapping(axons=256, neurons=32, count=999, widths=(4,))

    def test_codes_wider_than_the_type_are_refused_by_name(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                     platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True)
        wide = next(c for c in cores(program) if getattr(c, CORE_WEIGHT_BITS) == 4
                    and np.abs(np.asarray(c.get_core_matrix(program.ir))).max() > 1)
        setattr(wide, CORE_WEIGHT_BITS, 2)
        with pytest.raises(CoreTypeBindingError, match="binds to 2-bit cells"):
            check_core_codes(program.ir)


#: name -> (program builder, packed geometry): mixed-width programs, lanes L, E, P or none.
MIXED = {
    "residual_lanes": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                  platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True),
                       dict(axons=256, neurons=32, count=999)),
    "residual_no_lanes": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                     platform=lane_platform(None, **typed(LAST_STAGE), **GRIDS), hardware_bias=True),
                          dict(axons=256, neurons=32, count=999)),
    "residual_scheduling_allowed": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                               platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True),
                                    dict(axons=256, neurons=32, count=999, passes=128)),
    "lenet5_pool_on_two_bits": (
        lambda: build_lane_program(lenet5(), (1, 28, 28), num_classes=10, platform=lane_platform(**typed(
            [{"layers": ["classifier_*"], "weight_bits": 2}], widths=(4, 2), default=4, **GRIDS)), hardware_bias=True),
        dict(axons=1024, neurons=512, count=999)),
    "mlp_integer_scale": (lambda: build_lane_program(mlp(), (8,), num_classes=4, platform=lane_platform(**typed(
        [{"layers": ["_3"], "weight_bits": 3}], widths=(4, 3)))), dict(axons=256, neurons=16, count=999)),
}


@pytest.fixture(scope="module", params=sorted(MIXED))
def mixed(request):
    build, packed = MIXED[request.param]
    return request.param, build(), packed


class TestMixedWidthCertificates:
    def test_nf_equals_scm_equals_hcm_equals_the_golden_at_atol_zero(self, mixed):
        _name, program, packed = mixed
        x = real_inputs(program, 48)
        with torch.no_grad():
            scm = program.identity()(x)
            assert_exact(program.nf()(x), scm)
            widths = sorted({int(getattr(c, CORE_WEIGHT_BITS)) for c in cores(program)}, reverse=True)
            mapping = program.packed_mapping(**packed, widths=widths)
            hcm = ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu")(x)
        assert_exact(hcm, scm)
        codes, steps = run_golden(mapping, x.numpy())
        assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())

    def test_every_core_holds_codes_of_its_type_width_and_both_widths_are_used(self, mixed):
        _name, program, _packed = mixed
        assert check_core_codes(program.ir) == len(cores(program))
        assert len({int(getattr(c, CORE_WEIGHT_BITS)) for c in cores(program)}) == 2

    def test_the_narrow_type_changes_the_function(self, mixed):
        name, program, _packed = mixed
        if name != "residual_lanes":
            pytest.skip("one instance suffices")
        uniform = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                     platform=lane_platform(**GRIDS), hardware_bias=True)
        x = real_inputs(program, 16)
        with torch.no_grad():
            assert not torch.equal(program.identity()(x), uniform.identity()(x))


def test_a_multi_pass_program_on_typed_cores_is_refused_by_name():
    """The pass planner plans shape-only specs against one core budget: a multi-pass plan on typed cores is
    refused, never packed on a width-blind plan."""
    program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                 platform=lane_platform(**typed(LAST_STAGE), **GRIDS), hardware_bias=True)
    with pytest.raises(CoreTypeBindingError, match="multi-pass packing there is not built"):
        program.packed_mapping(axons=256, neurons=32, count=24, passes=128, widths=(4, 2))


def test_an_unbound_program_carries_no_width_anywhere():
    program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(**GRIDS),
                                 hardware_bias=True)
    assert not any(hasattr(n, CORE_WEIGHT_BITS) for n in program.ir.nodes if isinstance(n, NeuralCore))
    assert not any(hasattr(p, LAYER_WEIGHT_BITS) for p in program.model.get_perceptrons())
    mapping = program.packed_mapping(axons=256, neurons=32, count=999)
    assert not any(hasattr(hc, CORE_WEIGHT_BITS) for s in mapping.stages if s.kind == "neural"
                   for hc in s.hard_core_mapping.cores)


class TestRegistry:
    def test_the_rule_is_a_function_bearing_value_domain_passthrough_with_no_default(self):
        entry = REGISTRY[LAYER_RULE_KEY]
        assert entry.section == "platform_constraints" and entry.domain == "value" and entry.passthrough
        assert not entry.has_default() and entry.empty_means and LAYER_RULE_KEY in passthrough_keys()
        assert effective_class((LAYER_RULE_KEY,)) == FUNCTION_BEARING

    def test_a_core_type_width_is_function_bearing(self):
        assert effective_class(("cores.weight_bits",)) == FUNCTION_BEARING

    def test_the_rule_schema(self):
        schema = schema_of(REGISTRY[LAYER_RULE_KEY])
        assert value_errors(schema, LAST_STAGE, LAYER_RULE_KEY) == []
        assert value_errors(schema, [], LAYER_RULE_KEY)
        assert value_errors(schema, [{"layers": [], "weight_bits": 2}], LAYER_RULE_KEY)
        assert value_errors(schema, [{"layers": ["fc"], "weight_bits": 1}], LAYER_RULE_KEY)
        assert value_errors(schema, [{"layers": ["fc"], "weight_bits": 2, "x": 1}], LAYER_RULE_KEY)


class TestRecordPricing:
    def _typed_record(self, widths):
        from dataclasses import replace

        from unit.deployment_record.record_fixtures import make_full_record

        record = make_full_record()
        platform = dict(record.identity.platform)
        core = {"max_axons": 256, "max_neurons": 256, "count": 8}
        platform["cores"] = [dict(core, weight_bits=w) for w in widths]
        return replace(record, identity=replace(record.identity, platform=platform))

    def test_a_mixed_width_chip_withholds_the_one_width_and_names_the_per_bit_area(self):
        from mimarsinan.deployment_record.cost.absolute import price_absolute
        from mimarsinan.deployment_record.platform_physics import get_platform_physics
        from mimarsinan.deployment_record.quantities.from_record import from_record

        quantities = from_record(self._typed_record((4, 2)))
        assert not quantities.has("weight_bits")
        assert "weight widths [2, 4]" in str(quantities.withheld_reason("weight_bits"))
        pricing = price_absolute(quantities, get_platform_physics("generic_estimated_22nm"))
        area = next((t for t in pricing.terms if t.name == "chip_area_mm2"), None)
        named = (area.source if area is not None else "") + " ".join(r.reason for r in pricing.refusals)
        # [U-AREA] the generic's area_rule block (a type rule; its per-bit area constant dropped) names the withheld
        # widths beside the area it prices at the description's cell (R5 named area_per_cell_per_weight_bit unpriced)
        assert "weight bits withheld by the census" in named and "weight widths [2, 4]" in named

    def test_one_width_on_every_type_keeps_the_quantity(self):
        from mimarsinan.deployment_record.quantities.from_record import from_record

        assert from_record(self._typed_record((4, 4))).has("weight_bits")
