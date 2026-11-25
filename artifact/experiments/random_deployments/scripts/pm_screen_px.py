""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

TREE = sys.argv[1]
sys.path.insert(0, TREE + "/tests")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "11_wave2" / "final_units" / "ux" / "w13"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_rule as W

import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
try:
    from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
except ImportError as exc:
    raise SystemExit(f"REFUSED: RECORD_ABSENT: the head carries no record of the deployed arithmetic "
                     f"(mapping/ir/deployment_dtype.py, U-DAR): {exc}")
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.weight_programming import weight_programming_report
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

torch.set_num_threads(1)
PASSES = 4096
GEOMETRY = {"value_axons": [64, 128, 160, 256], "value_neurons": [16, 32], "lif_axons": 80, "lif_neurons": 16,
            "counts": [8, 16, 32, 64], "lif_widths": [4, 6, 8], "lif_layers": [1, 3]}


def build(ir, cores, multiplexed: bool, **caps):
    capabilities = ChipCapabilities(allow_scheduling=True, max_schedule_passes=PASSES,
                                    position_multiplexing=True if multiplexed else None, **caps)
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores, strategy=MappingStrategy.resolve(capabilities),
                                          max_schedule_passes=PASSES)


def neural(hybrid) -> int:
    return sum(1 for s in hybrid.stages if s.kind == "neural")


def _equal(a, b) -> bool:
    return bool(torch.equal(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)))


def value_rows(seed: int):
    import value_fuzz as VF
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs

    for offset in range(64):
        model, shape, classes, facts = VF.network(seed * 64 + offset)
        if facts["kind"] in ("conv", "residual"):
            break
    rng = random.Random(20_000 + seed)
    program = build_integer_program(model, shape, num_classes=classes, seed=seed)
    cores = [{"max_axons": rng.choice(GEOMETRY["value_axons"]), "max_neurons": rng.choice(GEOMETRY["value_neurons"]),
              "count": rng.choice(GEOMETRY["counts"]), "has_bias": True}]
    x = integer_inputs(program, 8, seed=seed + 1).double()
    with torch.no_grad():
        identity = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=program.ir), device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype)(x)
    row = dict(seed=seed, family="value", vehicle=facts, cores=cores,
               banks=len(getattr(program.ir, "weight_banks", {}) or {}))
    programs = {}
    for m in (False, True):
        try:
            programs[m] = build(program.ir, cores, m, allow_neuron_splitting=True, allow_coalescing=True)
        except ToolchainRefusal as exc:
            row[f"{'multiplexed' if m else 'replicated'}_refused"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    for m, hybrid in programs.items():
        name = "multiplexed" if m else "replicated"
        with torch.no_grad():
            got = ValueHybridCoreFlow(hybrid, device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype)(x)
        codes, steps = run_golden(hybrid, x.numpy())
        report = weight_programming_report(hybrid)
        row[name] = dict(passes=neural(hybrid), equal_identity=_equal(got, identity),
                         equal_golden=_equal(codes * steps, got), params_programmed=int(report.params_programmed),
                         params_unique=int(report.params_unique))
    return row


def lif_rows(seed: int):
    from unit.mapping.position_multiplexing import vehicles as PV

    rng = random.Random(30_000 + seed)
    widths = tuple(rng.choice(GEOMETRY["lif_widths"]) for _ in range(rng.randint(*GEOMETRY["lif_layers"])))
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    runs = {"streamed_lif": dict(spiking_mode="lif", cycle_accurate_lif_forward=True),
            "synchronized_lif": dict(spiking_mode="lif", cycle_accurate_lif_forward=True,
                                     lif_execution_synchronized=True)}


    if len({(W.cascade_of(r), W.armed(r)) for r in runs.values()}) != 1:
        raise SystemExit("REFUSED: WINDOW_FORMS: the two LIF runs take different window passes")
    with W.building(IRMapping, "map", runs["streamed_lif"]):
        ir = PV.lif_ir(PV.lif_conv_chain(seed=seed, widths=widths), GEOMETRY["lif_axons"], GEOMETRY["lif_neurons"])
    cores = [{"max_axons": GEOMETRY["lif_axons"], "max_neurons": GEOMETRY["lif_neurons"],
              "count": rng.choice(GEOMETRY["counts"]), "has_bias": True}]
    row = dict(seed=seed, family="lif", vehicle=dict(widths=list(widths)), cores=cores,
               banks=len(getattr(ir, "weight_banks", {}) or {}))

    def flow(hybrid, sem):
        return SpikingHybridCoreFlow(PV.LIF_SHAPE, hybrid, simulation_length=PV.T, thresholding_mode="<=", **sem)

    identity = build_identity_hybrid_mapping(ir_graph=ir)
    xs = [torch.rand(4, *PV.LIF_SHAPE, generator=torch.Generator().manual_seed(seed * 7 + b)) for b in range(3)]
    for m in (False, True):
        name = "multiplexed" if m else "replicated"
        try:
            hybrid = build(ir, cores, m, allow_neuron_splitting=True)
        except ToolchainRefusal as exc:
            row[f"{name}_refused"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        result = dict(passes=neural(hybrid), resident_passes=sum(bool(getattr(s, "schedule_weights_resident", False))
                                                                   for s in hybrid.stages if s.kind == "neural"))
        for run, sem in runs.items():
            with torch.no_grad():
                wants = [flow(identity, sem)(x) for x in xs]
                gots = [flow(hybrid, sem)(x) for x in xs]
            result[run] = "equal" if all(_equal(g, w) for g, w in zip(gots, wants)) else "DIFFERS"
            result[run + "_disc"] = max(float(w.std(0).max()) for w in wants) > 0
        row[name] = result
    return row


def main() -> int:
    out_path, s0, s1, family = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    produce = {"value": value_rows, "lif": lif_rows}[family]
    with open(out_path, "a") as out:
        for seed in range(s0, s1):
            t0 = time.time()
            try:
                row = produce(seed)
            except Exception as exc:
                row = dict(seed=seed, family=family, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            row["wall_s"] = round(time.time() - t0, 2)
            print(json.dumps(row), file=out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
