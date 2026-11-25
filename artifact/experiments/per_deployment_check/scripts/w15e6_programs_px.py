""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w15e1_programs_px as H

import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment, declaration_from_config
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir.core_law import graph_carries_laws

RA = Path("records/research_artifacts")
VALUE = dict(semantics="value_integer", window=1, pass_transfer="collapse",
             accumulation={"value_census": "float64", "int64_golden": "int64"}, upload={"value_census": "float32"})


def _facts(program) -> dict:
    neural = [s for s in program.stages if s.kind == "neural"]
    return {"passes": len(neural), "resident_passes": sum(bool(getattr(s, "schedule_weights_resident", False))
                                                         for s in neural),
            "pinned_passes": sum(bool(getattr(s, "schedule_resident_cores", None)) for s in neural),
            "hard_cores": sum(len(s.hard_core_mapping.cores) for s in neural)}


def _check(out, label: str, build: str, make, ir, decl_of, unbounded=None) -> None:
    ""
    row = {"vehicle": label, "build": build}
    try:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            program = make()
        row["compile_s"] = round(time.perf_counter() - t0, 3)
        row.update(_facts(program))
        report = check_deployment(program, ir_graph=ir, declaration=decl_of(program), compile_wall_s=row["compile_s"],
                                  unbounded_ir=unbounded)
        row.update(H._row_of(report))
        row["unbounded_ir"] = "given" if unbounded is not None else "not given (ir-tiling assumed)"
    except ToolchainRefusal as exc:
        row["refused"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        row["trace"] = traceback.format_exc()[-2000:]
    print(json.dumps(row, default=str), file=out, flush=True)
    print(label, build, row.get("theorem1") or row.get("refused") or row.get("error"), flush=True)


def _value(cores) -> CheckDeclaration:
    return CheckDeclaration(core_types=tuple(dict(c) for c in cores), **VALUE)


def lanes_r18(out, checkpoint: str, harness: str) -> None:
    import toolchain_probe_step2_px as L

    import e01_data
    from screen_models import CifarResNet18
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.export.law_solver import install_readout_laws
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
    from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
    from mimarsinan.mapping.support.grid_declarations import ParameterGrids
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    if L.sha256_file(Path(checkpoint)) != "e799b5c6aeb61763dd15026153baea5e8a4e35926813e488373d2ab1b3202089":
        raise SystemExit("REFUSED: checkpoint sha256 differs from the inventory")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module_state = {k: v for k, v in state.items() if not k.startswith("lane_quant.")}
    lanes = {}
    for k, v in state.items():
        if k.startswith("lane_quant.") and k.endswith(".act_theta"):
            name = k[len("lane_quant."): -len(".act_theta")]
            extra = state[f"lane_quant.{name}._extra_state"]
            lanes[name] = {"theta": float(v), "shift": float(state[f"lane_quant.{name}.act_shift"]),
                           "signed": bool(extra["signed"]), "calibrated": bool(extra["calibrated"])}
    lmap = L.lane_map(CifarResNet18(1.0, 100), lanes)
    images, labels = e01_data.load_train()
    train_idx, _val_idx = e01_data.e01_split(labels)
    cal = e01_data.inputs(images, e01_data.calibration_indices(train_idx))
    del images
    cfg = {"activation_bits": 4, "weight_bits": 4,
           "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 24, "output_law": "identity"},
           "weight_scale_rule": "fractional", "bias_resolution": "law",
           "onchip_arithmetic": {"classes": ["L", "E", "P"], "lanes_per_core": 8, "lane_block": "ng45_os_lane"},
           "core": {"max_axons": 256, "max_neurons": 512, "has_bias": True}, "allow_coalescing": True,
           "max_schedule_passes": 4096}
    work = Path(os.environ.get("E6_WORK", "/tmp")) / "lanes_r18_build"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        flow, ir, platform = L.build_program(module_state, 1.0, lanes, lmap, [cal[:1024], cal[1024:]], cfg, {}, work)
    build_s = round(time.perf_counter() - t0, 1)
    unbounded, note = None, ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            params = resolve_platform_mapping_params([dict(cfg["core"], count=1)], allow_coalescing=True)
            unbounded = IRMapping(q_max=quantization_bounds(4)[1], firing_mode="Default",
                                  hardware_bias=params.hardware_bias).map(flow.get_mapper_repr())
            install_boundary_grids(unbounded, flow, quantizer_state=platform["quantizer_state"])
            quantize_ir_graph(unbounded, 4, weight_quantization=True, grids=ParameterGrids.from_config(platform))
            install_readout_laws(unbounded, flow, platform, weight_quantization=True, weight_bits=4)
    except Exception as exc:
        unbounded, note = None, f"{type(exc).__name__}: {str(exc)[:200]}"
    print(json.dumps({"vehicle": "lanes R18 C1", "build_s": build_s, "ir_cores": len(ir.get_neural_cores()),
                      "ir_host_ops": len(ir.get_compute_ops()), "unbounded_ir_note": note or "built"}), file=out,
          flush=True)
    L._instrument_pool()
    with contextlib.redirect_stdout(io.StringIO()):
        discovery = build_hybrid_mapping_for_pipeline(ir, L.platform(cfg, 200000, False))
    count = L.taken_from_last_pool() or sum(len(s.hard_core_mapping.cores) for s in discovery.stages
                                            if s.kind == "neural")
    del discovery
    resident = [dict(cfg["core"], count=count)]
    _check(out, "lanes R18 C1", f"B0 resident ({count} cores)",
           lambda: build_hybrid_mapping_for_pipeline(ir, L.platform(cfg, count, False)), ir,
           lambda _p: _value(resident), unbounded)
    small = int(math.ceil(count / 8))
    scheduled = [dict(cfg["core"], count=small)]
    _check(out, "lanes R18 C1", f"B2 scheduled ({small} cores)",
           lambda: build_hybrid_mapping_for_pipeline(ir, L.platform(cfg, small, True)), ir,
           lambda _p: _value(scheduled), unbounded)


VIT_CORE, FRACTIONS = (768, 768), (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def _vit_lite(attention: str, depth: int = 7, seed: int = 7):
    from mimarsinan.models.families.vit_lite import ViTLiteCCT

    torch.manual_seed(seed)
    model = ViTLiteCCT(num_classes=10, img_size=32, patch=4, dim=256, depth=depth, heads=4, mlp_ratio=2.0,
                       attention=attention).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if not bool(parameter.abs().sum()):
                parameter.copy_(0.02 * torch.randn(parameter.shape, generator=generator))
    return model


def _vit_cores(count: int, bits: int = 8):
    return [{"max_axons": VIT_CORE[0], "max_neurons": VIT_CORE[1], "count": int(count),
             "operand": {"payload_bits": bits, "write_source": "host"}}]


def _vit_pack(ir, cores, scheduled: bool):
    from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=scheduled, max_schedule_passes=4096))
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores, strategy=strategy, max_schedule_passes=4096)


def vit(out, form: str) -> None:
    from unit.chip_simulation.value_run.operand_programs import build_operand_program, operand_platform

    with contextlib.redirect_stdout(io.StringIO()):
        program = build_operand_program(_vit_lite(form), (3, 32, 32), num_classes=10,
                                        platform=operand_platform(payload_bits=8, cores=_vit_cores(10 ** 6)))
    flat_cores = _vit_cores(10 ** 6)
    _check(out, f"ViT-Lite-7/4 {form}", "flat (unbounded counts)", lambda: _vit_pack(program.ir, flat_cores, False),
           program.ir, lambda _p: _value(flat_cores), program.ir)
    with contextlib.redirect_stdout(io.StringIO()):
        flat = _vit_pack(program.ir, flat_cores, False)
    widest = max(len(s.hard_core_mapping.cores) for s in flat.stages if s.kind == "neural")
    del flat
    for fraction in FRACTIONS:
        budget = max(1, min(widest - 1, math.ceil(widest * fraction)))
        cores = _vit_cores(budget)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _vit_pack(program.ir, cores, True)
        except (RuntimeError, ValueError) as refusal:
            print(json.dumps({"vehicle": f"ViT-Lite-7/4 {form}", "budget": budget,
                              "refused": f"{type(refusal).__name__}: {str(refusal).splitlines()[0][:160]}"}),
                  file=out, flush=True)
            continue
        _check(out, f"ViT-Lite-7/4 {form}", f"scheduled ({budget} cores, {fraction} of the widest segment's {widest})",
               lambda: _vit_pack(program.ir, cores, True), program.ir, lambda _p: _value(cores), program.ir)
        break


def _template(cell: str, gain: float, platform_of, plan=None):
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.mapping.residency.executed import templates as TV

    vehicle = TV.template_vehicle(cell)
    bits = int(vehicle.config.get("weight_bits", 8))
    TV.activate(vehicle.ir, gain, bits=bits)
    for op in vehicle.ir.get_compute_ops():
        module = (op.params or {}).get("module")
        if isinstance(module, torch.nn.Module):
            module.eval()
    platform = {**platform_of(TV, vehicle), "weight_bits": bits}
    if plan is not None:
        platform["residency_plan"] = dict(plan)

    def decl(_program):
        return declaration_from_config(vehicle.config, cores=platform["cores"], has_laws=graph_carries_laws(vehicle.ir),
                                       backends=H._backends(vehicle.config))

    return (lambda: build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)), \
        vehicle.ir, decl


def mux(out) -> None:
    from unit.mapping.position_multiplexing import vehicles as M

    with contextlib.redirect_stdout(io.StringIO()):
        resnet = M.integer_resnet()
    cores = [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}]
    _check(out, "integer CIFAR ResNet (8, 16)", "position multiplexed 160 x 32 x 64",
           lambda: M.build(resnet.ir, cores, True), resnet.ir, lambda _p: _value(cores), resnet.ir)

    def multiplexed(TV, vehicle):
        platform = TV.residency_platform(vehicle, 1.0, None)
        platform.update(max_schedule_passes=16384, position_multiplexing=True)
        return platform

    make, ir, decl = _template("t0_49", 24.0, multiplexed)
    _check(out, "tier-0 t0_49 (streamed LIF)", "position multiplexed, its own cores", make, ir, decl)


def w03(out) -> None:
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, resnet18_quarter

    cores = [{"max_axons": 1153, "max_neurons": 128, "count": 400, "has_bias": False}]
    platform = {"cores": cores, "weight_bits": 8, "core_semantics": "mvm", "activation_bits": 8,
                "allow_scheduling": True, "allow_coalescing": False, "allow_neuron_splitting": False,
                "max_schedule_passes": 32}
    pmap = resolve_platform_mapping_params(cores, allow_coalescing=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r18 = build_integer_program(resnet18_quarter(), (3, 32, 32), num_classes=10,
                                    max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons)
    for plan in ({"step": "floor"}, {"step": "stateless"}):
        _check(out, "CIFAR ResNet-18 w1/4", f"residency_plan {plan}",
               lambda plan=plan: build_hybrid_mapping_for_pipeline(r18.ir, {**platform, "residency_plan": plan}),
               r18.ir, lambda _p: _value(cores))
    make, ir, decl = _template("t0_03", 32.0, lambda TV, v: TV.residency_platform(v, 2.0, None))
    _check(out, "tier-0 t0_03 (synchronized LIF)", "x2 cores, the fit pins the unscheduled pack", make, ir, decl)
    make, ir, decl = _template("t0_03", 32.0, lambda TV, v: TV.residency_platform(v, 1.15, None),
                                plan={"step": "index", "index": 1})
    _check(out, "tier-0 t0_03 (synchronized LIF)", "x1.15 cores, residency_plan step 1", make, ir, decl)


def main() -> int:
    mode = sys.argv[1]
    if mode in ("configs", "archived", "fuzz", "r18", "lenet", "x3a"):
        return H.main()
    with open(sys.argv[2], "a") as out:
        if mode == "lanes_r18":
            lanes_r18(out, sys.argv[3], sys.argv[4])
        elif mode == "vit":
            vit(out, sys.argv[3])
        elif mode == "mux":
            mux(out)
        elif mode == "w03":
            w03(out)
        else:
            raise SystemExit(f"unknown mode {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
