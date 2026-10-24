"""WI-5e certificates: a lane program's NF = SCM = HCM = int64 golden at atol 0 on G2's instances, single and
multi-pass, and its lanes are on chip (no host op the declared classes absorb)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.ir.lane_law import LaneLaw
from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
from mimarsinan.mapping.lanes.install import LANE_REPORT
from unit.chip_simulation.value_run.integer_programs import (
    PRUNE,
    assert_exact,
    conv1d_net,
    lenet5,
    mlp,
    residual_block_net,
)
from unit.mapping.lanes.lane_programs import ALL, build_lane_program, cores, host_ops, lane_platform, real_inputs

#: name -> (builder, packed geometry, host ops left on the host by the lane subset L, E, P).
LANE_INSTANCES = {
    "mlp": (lambda: build_lane_program(mlp(), (8,), num_classes=4, platform=lane_platform()),
            dict(axons=256, neurons=5), []),
    "lenet5": (lambda: build_lane_program(lenet5(), (1, 28, 28), num_classes=10, platform=lane_platform()),
               dict(axons=1024, neurons=512), ["MaxPool2d", "MaxPool2d"]),
    "pruned_bank_conv": (lambda: build_lane_program(lenet5(), (1, 28, 28), num_classes=10, prune=PRUNE,
                                                    platform=lane_platform()),
                         dict(axons=1024, neurons=512), ["MaxPool2d", "MaxPool2d"]),
    "residual_block": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                  platform=lane_platform()),
                       dict(axons=256, neurons=32), []),
    "residual_block_bias_rows": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                            platform=lane_platform(), hardware_bias=True),
                                 dict(axons=256, neurons=32), []),
    "conv1d": (lambda: build_lane_program(conv1d_net(), (2, 16), num_classes=5, platform=lane_platform()),
               dict(axons=128, neurons=16), []),
    "pruned_multi_pass": (lambda: build_lane_program(lenet5(), (1, 28, 28), num_classes=10, prune=PRUNE,
                                                     platform=lane_platform()),
                          dict(axons=1024, neurons=256, count=3, passes=128), ["MaxPool2d", "MaxPool2d"]),
    "residual_block_multi_pass": (lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                             platform=lane_platform()),
                                  dict(axons=256, neurons=32, count=24, passes=128), []),
}


@pytest.fixture(scope="module", params=sorted(LANE_INSTANCES))
def instance(request):
    build, packed, host = LANE_INSTANCES[request.param]
    return request.param, build(), packed, host


def _run(flow, x):
    with torch.no_grad():
        return flow(x)


class TestLaneCertificates:
    def test_nf_equals_scm_equals_hcm_equals_the_golden_at_atol_zero(self, instance):
        _name, program, packed, _host = instance
        x = real_inputs(program, 48)
        scm = _run(program.identity(), x)
        assert_exact(_run(program.nf(), x), scm)
        mapping = program.packed_mapping(**packed)
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        hcm = _run(ValueHybridCoreFlow(mapping, dtype=torch.float64), x)
        assert_exact(hcm, scm)
        golden_codes, golden_steps = run_golden(mapping, x.numpy())
        assert np.array_equal(golden_codes * golden_steps, hcm.reshape(golden_codes.shape).numpy())

    def test_only_the_ops_outside_the_lane_subset_stay_on_the_host(self, instance):
        _name, program, _packed, host = instance
        assert host_ops(program) == host

    def test_every_core_carries_a_lane_law_and_the_fusion_report(self, instance):
        _name, program, _packed, _host = instance
        assert all(isinstance(c.readout_law, LaneLaw) for c in cores(program))
        report = getattr(program.ir, LANE_REPORT)
        assert report["classes"] == sorted(ALL) and report["clamps"] >= 1


class TestScheduledCertificate:
    def test_a_scheduled_lane_program_certifies_its_pass_carries(self):
        """A lane program is one neural segment, so its passes carry values the identity program keeps inside the
        segment: the C-edge exposes them on the identity program and compares them, instead of refusing them as
        backend-only nodes."""
        from mimarsinan.certification.value_certificate import certify_twin_flow_values, exposed_nodes
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
        from mimarsinan.torch_mapping.value_parity.value_program import fp64_identity_program

        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        mapping = program.packed_mapping(axons=256, neurons=32, count=24, passes=128)
        backend = ValueHybridCoreFlow(mapping, dtype=torch.float64)
        x = real_inputs(program, 8)
        plain, detail = certify_twin_flow_values(fp64_identity_program(program.ir), backend, x, atol=0.0)
        assert not plain.passed and "backend-only=[]" not in detail, "the carries are invisible to the plain identity"
        carried = exposed_nodes(mapping)
        cert, detail = certify_twin_flow_values(fp64_identity_program(program.ir, exposed=carried), backend, x,
                                                atol=0.0)
        assert cert.passed and detail.endswith("backend-only=[]") and "reference-only=[] " in detail, detail
        assert cert.neuron_windows_compared > plain.neuron_windows_compared
        resident = program.packed_mapping(axons=256, neurons=32)
        assert len(exposed_nodes(resident)) == 1, "a resident lane program writes the program output only"


class TestLaneForms:
    def test_a_clamp_lane_writes_the_unsigned_2b_code_grid(self):
        program = build_lane_program(mlp(), (8,), num_classes=4, platform=lane_platform())
        clamps = [c for c in cores(program) if int(c.readout_law.lo.min()) == 0]
        assert clamps and all(int(c.readout_law.hi.max()) == 15 for c in clamps)
        model = program.nf()
        seen = []
        quantizers = [p.scaler for p in model.get_perceptrons() if getattr(p.scaler, "lane_unsigned", False)]
        def hook(module, _args, out):
            if not module._state.rerunning:  # the stamped law's own accumulate reruns the perceptron
                seen.append(out / (module.range / 15.0))

        handles = [q.register_forward_hook(hook) for q in quantizers]
        try:
            _run(model, real_inputs(program, 64))
        finally:
            for handle in handles:
                handle.remove()
        codes = torch.cat([c.flatten() for c in seen])
        assert len(quantizers) == len(clamps) == 2
        assert float((codes - torch.round(codes)).abs().max()) < 1e-9, "every lane value is a code of its grid"
        codes = torch.round(codes)
        assert float(codes.min()) == 0.0 and float(codes.max()) == 15.0, "both ends of the 16-code grid are read"

    def test_a_fused_add_gives_its_accumulator_one_operand_row_per_column(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        fused = [c for c in cores(program) if getattr(c, OPERAND_ROWS, None) is not None]
        assert fused
        for core in fused:
            rows = np.asarray(getattr(core, OPERAND_ROWS))
            matrix = np.asarray(core.get_core_matrix(program.ir))
            assert rows.size == core.get_output_count() and bool((rows >= 0).all())
            assert not matrix[rows, :].any(), "an operand row carries no crossbar weight"
            law = core.readout_law
            assert bool((law.operand_m > 0).all()) and np.array_equal(law.operand_row, rows)

    def test_the_pool_lane_is_a_requantizing_reduction(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        pools = [c for c in cores(program) if c.perceptron_index is None]
        assert pools and all(float(c.threshold) == 64.0 for c in pools)  # 8 x 8 window
        assert all(set(np.unique(np.asarray(c.core_matrix))) <= {0, 1} for c in pools)
        report = getattr(program.ir, LANE_REPORT)
        assert report["pools"] == 1 and report["pool_cores"] == len(pools)

    @pytest.mark.parametrize("classes,host", [
        # Without E an add stays on the host, and with it the ReLU after it and the pool after that (a lane's
        # producers must be on chip).
        (("L",), ["_operator.add", "ReLU", "_operator.add", "ReLU", "AdaptiveAvgPool2d"]),
        (("L", "P"), ["_operator.add", "ReLU", "_operator.add", "ReLU", "AdaptiveAvgPool2d"]),
        (("L", "E"), ["AdaptiveAvgPool2d"]),
    ])
    def test_each_class_absorbs_its_own_ops_only(self, classes, host):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                     platform=lane_platform(classes))
        assert host_ops(program) == host
        x = real_inputs(program, 16)
        assert_exact(_run(program.nf(), x), _run(program.identity(), x))

    def test_the_off_ablation_is_the_lane_free_program(self):
        from unit.mapping.value_grid_identity import _ir_digest

        plain = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(None))
        off = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform([]))
        assert _ir_digest(off.ir) == _ir_digest(plain.ir)
        x = real_inputs(plain, 8)
        assert torch.equal(_run(off.identity(), x), _run(plain.identity(), x))

    def test_lanes_declared_on_the_float_path_change_nothing(self):
        from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed

        platform = {"activation_bits": 4, "onchip_arithmetic": {"classes": ["L", "E", "P"]}}
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(None))
        assert lane_absorbed(program.model, platform) == frozenset()
        assert install_planned_lanes(program.model, platform, 4) is None

    def test_a_float32_model_reads_the_lane_codes_of_its_float64_twin(self):
        """The pipeline runs the stamped model in float32: every lane end (the pool lane's included) reads its input
        on its grid within float32 rounding and writes the float64 law's values."""
        from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of

        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        first = next(iter(program.model.get_perceptrons()))
        grid = boundary_grid_of(first.input_activation)
        codes = torch.randint(-grid.levels, grid.levels + 1, (8, 3, 16, 16), generator=torch.Generator().manual_seed(4))
        x64 = codes.double() * float(grid.step)  # on the entry grid: no tie for either precision's snap
        reference = _run(program.model.double(), x64)
        assert torch.equal(_run(program.model.float(), x64.float()), reference.float())

    def test_lanes_change_the_function(self):
        lanes = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        plain = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(None))
        x = real_inputs(lanes, 16)
        assert not torch.equal(_run(lanes.identity(), x), _run(plain.identity(), x))
        assert len(host_ops(plain)) == 8 and host_ops(lanes) == []


def _declared_r18(tmp_path):
    """ResNet-18 w1/4 on the lane chip with every Wave 2 declaration: the declared map (an offset floor encoder at
    the stem, a range at every lane end, an offset on each projection lane), the fractional weight scale, the bias
    at the law's resolution and the identity output law."""
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from unit.mapping.lanes.lane_programs import write_state

    platform = lane_platform(weight_scale_rule="fractional", bias_resolution="law")
    platform["readout_law"]["output_law"] = "identity"

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        fx_of = {id(p): name for name, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
        document = {fx_of[id(entries[0])]: {"entry": {"scale": 6.0, "offset": 3.0, "rounding": "floor",
                                                      "codes": [0, 15]}}}
        for perceptron, quantizer in zip(perceptrons[:-1], quantizers[:-1]):
            body = document.setdefault(fx_of[id(perceptron)], {})
            body["readout_range"] = float(quantizer.range) * 0.9
            if "shortcut" in fx_of[id(perceptron)]:
                body["readout_offset"] = 0.25 * float(quantizer.range)
        pools = {name: {"readout_range": float(q.range) * 0.8} for name, q in lanes}
        platform["quantizer_state"] = write_state(tmp_path / "state.json", document, pools)

    from unit.chip_simulation.value_run.integer_programs import resnet18_quarter

    return build_lane_program(resnet18_quarter(), (3, 32, 32), num_classes=10, platform=platform,
                              hardware_bias=True, state_hook=declare)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_resnet18_quarter_lane_program_certifies_single_and_multi_pass(tmp_path):
    program = _declared_r18(tmp_path)
    assert host_ops(program) == []
    report = getattr(program.ir, LANE_REPORT)
    assert (report["clamps"], report["adds"], report["pools"]) == (17, 8, 1)
    offsets = {round(float(c.readout_law.output_offset[0]), 9) for c in cores(program)}
    assert len(offsets) > 1, "the projection lanes write offset grids"
    x = real_inputs(program, 6)
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

    scm = _run(program.identity(), x)
    assert_exact(_run(program.nf(), x), scm)
    for geometry in (dict(axons=256, neurons=256, coalescing=True, count=99999),
                     dict(axons=1280, neurons=256, count=8, passes=1024)):
        mapping = program.packed_mapping(**geometry)
        assert_exact(_run(ValueHybridCoreFlow(mapping, dtype=torch.float64), x), scm)
        codes, steps = run_golden(mapping, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), scm)
