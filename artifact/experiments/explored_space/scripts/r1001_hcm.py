""
from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import sys
import time

WT = sys.argv[1]
sys.path.insert(0, os.path.join(WT, "src"))
sys.path.insert(0, os.path.join(WT, "tests"))


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def main() -> int:
    out_path = sys.argv[2]
    argv = sys.argv[3:]

    def opt(name, default):
        return type(default)(argv[argv.index(name) + 1]) if name in argv else default

    cores, axons, neurons = opt("--cores", 0), opt("--axons", 256), opt("--neurons", 256)
    capacity_row = opt("--capacity-row", "")
    import torch
    from mimarsinan.models.builders import BUILDERS_REGISTRY
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.chip_simulation.value_run.integer_programs import build_integer_program

    record = {"member": "CIFAR ResNet-1001 (pre-activation bottleneck, 111 blocks per stage)", "stages": {},
              "geometry": {"axons": axons, "neurons": neurons, "count": cores, "weight_bits": 8},
              "class": "MEASURED walls and peak RSS, single run, provisional"}
    t0 = time.time()
    torch.manual_seed(0)
    model = BUILDERS_REGISTRY["cifar_resnet_family"]("cpu", (3, 32, 32), 10, {}).build(
        {"blocks_per_stage": 111, "block_kind": "bottleneck", "block_order": "pre_activation"}).eval()
    record["stages"]["build"] = {"wall_s": round(time.time() - t0, 1), "peak_rss_gib": round(rss_gib(), 2),
                                 "params": int(sum(p.numel() for p in model.parameters()))}
    t1 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            program = build_integer_program(model, (3, 32, 32), num_classes=10, max_axons=axons, max_neurons=neurons,
                                            allow_coalescing=True)
    except Exception as exc:
        record["stages"]["value_program"] = {"wall_s": round(time.time() - t1, 1), "peak_rss_gib": round(rss_gib(), 2),
                                             "refusal": {"name": type(exc).__name__, "reason": str(exc)[:400]}}
        with open(out_path, "w") as handle:
            json.dump(record, handle, indent=1, default=str)
        return 1
    record["stages"]["value_program"] = {"wall_s": round(time.time() - t1, 1), "peak_rss_gib": round(rss_gib(), 2),
                                         "ir_cores": len(program.ir.get_neural_cores())}
    print(json.dumps(record), flush=True)
    if not cores:
        cores = len(program.ir.get_neural_cores())
        record["geometry"]["count"] = cores
        record["geometry"]["count_rule"] = "one hard core per IR neural core (resident, allow_scheduling false)"
    if cores:
        platform = {"cores": [{"max_axons": axons, "max_neurons": neurons, "count": cores, "has_bias": False}],
                    "weight_bits": 8, "core_semantics": "mvm", "activation_bits": 8, "allow_scheduling": False,
                    "allow_coalescing": True, "allow_neuron_splitting": False}
        t2 = time.time()
        try:
            hybrid = build_hybrid_mapping_for_pipeline(program.ir, platform)
            stages = list(getattr(hybrid, "stages", []))
            record["stages"]["hard_core_mapping"] = {
                "wall_s": round(time.time() - t2, 1), "peak_rss_gib": round(rss_gib(), 2),
                "neural_stages": sum(1 for s in stages if getattr(s, "kind", None) == "neural"),
                "hard_cores_used": sum(len(getattr(s.hard_core_mapping, "cores", []))
                                       for s in stages if getattr(s, "kind", None) == "neural")}
        except Exception as exc:
            record["stages"]["hard_core_mapping"] = {"wall_s": round(time.time() - t2, 1),
                                                     "peak_rss_gib": round(rss_gib(), 2),
                                                     "refusal": {"name": type(exc).__name__, "reason": str(exc)[:400]}}
    if capacity_row and os.path.exists(capacity_row):
        for line in open(capacity_row):
            row = json.loads(line)
            if row.get("network") == "resnet1001_preact" and row.get("x_fluid") is not None:
                record.setdefault("shape_only_points", []).append(
                    {k: row.get(k) for k in ("target", "cores", "softcores", "packed_tiles", "x_fluid", "x_packed",
                                             "layout_wall_s", "peak_rss_bytes")})
    record["total_wall_s"] = round(time.time() - t0, 1)
    with open(out_path, "w") as handle:
        json.dump(record, handle, indent=1, default=str)
    print(json.dumps(record, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
