""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import import_gate_checkpoint as ig


from mimarsinan.certification import operand_certificate as vc
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

GEOMETRY = (768, 768)
FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def cores(count: int, bits) -> list:
    core = {"max_axons": GEOMETRY[0], "max_neurons": GEOMETRY[1], "count": int(count)}
    if bits is not None:
        core["operand"] = {"payload_bits": int(bits), "write_source": "host"}
    return [core]


def pack(ir, cores_config, scheduled: bool):
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=scheduled, max_schedule_passes=4096))
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config, strategy=strategy,
                                          max_schedule_passes=4096)


def segment_tiles(mapping) -> list:
    ""
    return [len(s.hard_core_mapping.cores) for s in mapping.stages if s.kind == "neural"]


def program_shape(mapping) -> dict:
    banks = getattr(mapping, "dynamic_bank_bindings", {}) or {}
    per_segment, operand_tiles, static_tiles, write_regions = {}, 0, 0, 0
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        per_segment[stage.schedule_segment_index] = per_segment.get(stage.schedule_segment_index, 0) + 1
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            holds = [p for p in placements if p.get("weight_bank_id") in banks]
            write_regions += len(holds)
            if holds:
                operand_tiles += 1
            else:
                static_tiles += 1
    return {"segments": len(per_segment), "neural_stages": sum(per_segment.values()),
            "max_passes_per_segment": max(per_segment.values()) if per_segment else 0,
            "multi_pass_segments": sum(1 for v in per_segment.values() if v > 1),
            "tiles_static": static_tiles, "tiles_operand": operand_tiles, "operand_regions": write_regions,
            "host_stages": sum(1 for s in mapping.stages if s.kind == "compute")}


def build_packed(program, bits):
    flat = pack(program.ir, cores(10 ** 6, bits), scheduled=False)
    widest = max(segment_tiles(flat))
    attempts, packed, budget = [], None, 0
    for fraction in FRACTIONS:
        budget = max(1, min(widest - 1, math.ceil(widest * fraction)))
        try:
            packed = pack(program.ir, cores(budget, bits), scheduled=True)
            attempts.append({"fraction": fraction, "cores": budget, "built": True})
            break
        except (RuntimeError, ValueError) as refusal:
            attempts.append({"fraction": fraction, "cores": budget, "built": False,
                             "refusal": f"{type(refusal).__name__}: {str(refusal).splitlines()[0][:200]}"})
    if packed is None:
        raise SystemExit(f"REFUSED: no declared budget builds: {attempts}")
    return packed, {"geometry": list(GEOMETRY), "tiles_unbounded": sum(segment_tiles(flat)),
                    "widest_segment_tiles": widest, "cores": budget, "attempts": attempts,
                    "cores_declared": cores(budget, bits), "packed": program_shape(packed)}


class OutputCapture:
    ""

    def __init__(self):
        self.outputs = []
        self._inner = vc._flow_outputs

    def __enter__(self):
        def capture(mapping, samples):
            flow, out, state = self._inner(mapping, samples)
            self.outputs.append(out.detach().cpu().numpy().copy())
            return flow, out, state
        vc._flow_outputs = capture
        return self

    def __exit__(self, *exc):
        vc._flow_outputs = self._inner


def run_count(program, packed, data, batch: int, start: int, stop: int) -> dict:
    hcm = ValueHybridCoreFlow(packed, device="cpu", dtype=torch.float64)
    nf = program.model.double().eval()
    pred_hcm, pred_nf, differing = [], [], 0
    for b in range(start, stop):
        x = ig.normalize64(data.val[b * batch:(b + 1) * batch])
        if not len(x):
            break
        with torch.no_grad():
            got, want = hcm(x), nf(x)
        differing += int(((got - want).abs() > 0).any(dim=-1).sum())
        pred_hcm.append(got.argmax(-1).numpy())
        pred_nf.append(want.argmax(-1).numpy())
        print(f"[count] batch {b}: {time.strftime('%H:%M:%S')}", flush=True)
    return {"pred_hcm": np.concatenate(pred_hcm).tolist(), "pred_nf": np.concatenate(pred_nf).tolist(),
            "images_with_any_logit_differing_hcm_nf": differing}


def run_certify(program, packed, data, batch: int, start: int, stop: int) -> dict:
    identity = program.identity_mapping()
    nf = program.model.double().eval()
    rows, preds = [], []
    for b in range(start, stop):
        x = ig.normalize64(data.val[b * batch:(b + 1) * batch])
        if not len(x):
            break
        t0 = time.time()
        with OutputCapture() as capture:
            certificate = vc.certify_operand_program(nf, identity, packed, x)
        scm_out, hcm_out = capture.outputs[0], capture.outputs[1]
        preds.append(hcm_out.argmax(-1))
        rows.append({"batch": b, "images": int(len(x)), "passed": bool(certificate.passed),
                     "edges": [vars(e) for e in certificate.edges], "payload": vars(certificate.payload),
                     "nonzero_census": certificate.nonzero_census,
                     "scm_hcm_argmax_equal": bool((scm_out.argmax(-1) == hcm_out.argmax(-1)).all()),
                     "wall_s": round(time.time() - t0, 1)})
        print(f"[certify] batch {b}: {certificate.summary()} ({rows[-1]['wall_s']} s)", flush=True)
    return {"rows": rows, "pred_hcm": np.concatenate(preds).tolist() if preds else []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("outdir")
    parser.add_argument("mode", choices=("count", "certify"))
    parser.add_argument("--batches", default="0:79")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start, stop = (int(v) for v in args.batches.split(":"))
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = ig.load_gate_data()
    program = ig.build_gate_program(args.key, data)
    t_build = time.time() - t0
    _job, _form, arm, bits = ig.JOBS[args.key]
    packed, packing = build_packed(program, bits if arm == "operand" else None)
    t_pack = time.time() - t0 - t_build
    body = run_count if args.mode == "count" else run_certify
    result = body(program, packed, data, args.batch, start, stop)
    labels = data.labels[start * args.batch:stop * args.batch]
    pred = np.asarray(result["pred_hcm"])
    record = {
        "key": args.key, "job": ig.JOBS[args.key], "mode": args.mode, "batches": [start, stop],
        "batch": args.batch, "images": int(len(pred)), "correct_hcm": int((pred == labels[:len(pred)]).sum()),
        "split_sha256": data.split_sha256, "calibration_sha256": data.calibration_sha256,
        "ir": {"cores": len(program.ir.get_neural_cores()), "host_ops": len(program.ir.get_compute_ops()),
               "banks": len(program.ir.dynamic_bank_bindings),
               "payload_cells_per_inference": int(sum(b.cells for b in program.ir.dynamic_bank_bindings.values()))},
        "lowered": [vars(p) for p in program.report.lowered], "host_products": [vars(p) for p in program.report.host],
        "packing": packing, "provenance": program.provenance, "import_map": program.import_map,
        "threads": args.threads, "torch": torch.__version__,
        "wall_s": {"build": round(t_build, 1), "pack": round(t_pack, 1), "total": round(time.time() - t0, 1)},
        **result,
    }
    if "pred_nf" in result:
        record["correct_nf"] = int((np.asarray(result["pred_nf"]) == labels[:len(pred)]).sum())
    name = f"{args.key}.{args.mode}.{start:03d}-{stop:03d}.json"
    (out_dir / name).write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record[k] for k in ("key", "mode", "images", "correct_hcm", "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
