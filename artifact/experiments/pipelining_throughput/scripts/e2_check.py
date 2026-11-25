""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/work/worktrees/throughput")
_spec = importlib.util.spec_from_file_location("w0_price_e1c", HERE / "w0_price.py")
W0 = importlib.util.module_from_spec(_spec)
sys.modules["w0_price_e1c"] = W0
_spec.loader.exec_module(W0)
WP, W48 = W0.WP, W0.W48


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=("r18_224", "c1"))
    ap.add_argument("--cores", type=int, default=256)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--priced", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--inputs", type=int, default=None)
    args = ap.parse_args(argv)
    start = time.time()
    args.inputs = args.inputs or args.slots
    out = {"format": "thru-e2-check/1", "slots": args.slots, "inputs": args.inputs, "branch": "tcad/throughput-replication", "net": args.net,
           "head": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(), "cores": args.cores, "limit": args.limit,
           "script_sha256": W48.sha256(Path(__file__).resolve()), "stages": []}

    def stage(name, **facts):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        out["stages"].append({"stage": name, "t_s": round(time.time() - start, 1), "peak_gb": round(peak, 2), **facts})
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
        print(json.dumps(out["stages"][-1], default=str), flush=True)

    tc = W0.W0Toolchain(WT)
    tc.count = args.cores
    sys.path.insert(0, str(WT / "tests"))
    import torch
    from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
    from mimarsinan.deployment_record.build.pass_carry import candidate_carry_census
    from mimarsinan.deployment_record.build.program_census import mapping_program_passes
    from mimarsinan.deployment_record.build.write_census import period_census
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
    from mimarsinan.mapping.support.schedule.pass_carry import pass_carry_census
    from mimarsinan.mapping.support.wavefront_stages import stage_groups
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.certification.program_check.unit_vehicles import VALUE
    from unit.chip_simulation.value_run.integer_programs import integer_inputs
    from unit.mapping.lanes.lane_planes import STEPS, TRANSFER, ladder, network_of, period_rows
    from unit.mapping.lanes.lane_programs import build_lane_program

    src = WP.with_pipelined_arm(W48.route_members())[W0.SOURCE_MEMBER]
    declare = {"max_schedule_passes": W0.WIDE, "pipelined_multiplexing": True, "layer_replication": True,
               "wavefront_group_cores": args.limit, "batch_pipeline_slots": args.slots,
               "carry_input_slots": tc.chip_values(W0.CHIP).get("carry_input_slots")}
    platform = tc.platform(src["arms"]["B5m"], declare=declare, k=args.slots)
    out["platform_keys"] = {k: platform.get(k) for k in ("cores", "layer_replication", "wavefront_group_cores",
                                                         "pipelined_multiplexing", "max_schedule_passes",
                                                         "activation_bits", "weight_bits", "onchip_arithmetic",
                                                         "readout_law", "allow_coalescing", "batch_pipeline_slots",
                                                         "carry_input_slots", "inference_batch_K")}
    if args.net == "r18_224":
        import torchvision


        out["dropped"] = {"quantizer_state": platform.pop("quantizer_state", None) is not None}
        shape, classes = (3, 224, 224), 1000
        torch.manual_seed(0)
        model = torchvision.models.resnet18(num_classes=classes).eval()
    else:
        shape, classes = (3, 32, 32), 100
        torch.manual_seed(0)
        model = tc.members_mod.build_x10ta(src["depth"], src["width"], classes).eval()
    params = resolve_platform_mapping_params(platform["cores"], allow_coalescing=bool(platform.get("allow_coalescing")))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        program = build_lane_program(model, shape, num_classes=classes, platform=dict(platform),
                                     weight_bits=int(platform["weight_bits"]), max_axons=params.effective_max_axons,
                                     max_neurons=params.effective_max_neurons, hardware_bias=params.hardware_bias,
                                     allow_coalescing=params.allow_coalescing)
    stage("lane program", ir_nodes=len(program.ir.nodes))


    network = network_of(program, classes)
    census = LD.shape_census(network, platform)
    scheduled = LD.schedule_program(network, census, platform, ladder().program_facts,
                                    tiles=LD.packed_tiles(census, platform))
    if scheduled.refusal is not None:
        stage("candidate plane refused", refusal=scheduled.refusal)
        return 1
    stage("candidate plane", passes=len(scheduled.facts.passes.rows))
    mapping = build_hybrid_mapping_for_pipeline(program.ir, platform)
    neural = [s for s in mapping.stages if s.kind == "neural"]
    stage("built program", stages=len(mapping.stages), neural=len(neural))
    cap = {(int(c["max_axons"]), int(c["max_neurons"])): int(c["count"]) for c in platform["cores"]}
    bits = int(platform["weight_bits"])
    cand_rows, built_rows = period_rows(scheduled.facts.passes), period_rows(mapping_program_passes(mapping))
    cand_carry = candidate_carry_census(census.softcores, scheduled.noc, timesteps=STEPS, pass_transfer=TRANSFER)
    built_carry = pass_carry_census(mapping.stages, STEPS, TRANSFER, groups=stage_groups(mapping.stages))

    def written_bytes(passes):
        return sum(p.payload_bytes for p in period_census(passes, weight_bits=bits, capacity=cap).census.passes)
    census_cmp = {
        "passes": [cand_rows["passes"], built_rows["passes"]],
        "barriers (schedule census)": [list(scheduled.noc.schedule_census),
                                       list(scheduled.facts.programming.schedule_census)],
        "written bytes per period": [written_bytes(scheduled.facts.passes), written_bytes(mapping_program_passes(mapping))],
        "carry": [cand_carry, built_carry]}
    same = {k: v[0] == v[1] for k, v in census_cmp.items()}
    for key in ("tiles", "cells", "written"):
        same[f"per pass {key}"] = cand_rows[key] == built_rows[key]
    same["barriers (schedule census)"] = (census_cmp["barriers (schedule census)"][0]
                                          == census_cmp["barriers (schedule census)"][1]
                                          and census_cmp["barriers (schedule census)"][0][0] == built_rows["passes"])
    out["census"] = {"compared": census_cmp, "equal": same}
    if args.priced is not None and args.priced.exists():
        d = json.loads(args.priced.read_text())
        cell = d["members"][W0.MEMBER]["arms"][d["arm"]]["candidate"][f"study:shipped|K{args.slots}"]
        out["census"]["pool_cell"] = {"file": str(args.priced), "passes": cell["passes"], "carry": cell["carry"],
                                      "written_passes": cell.get("written_passes"),
                                      "reprogram_passes": cell.get("reprogram_passes")}

        from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
        steps = platform.get("simulation_steps")
        at_row = pass_carry_census(mapping.stages, steps, transfer_for(streamed=False), groups=stage_groups(mapping.stages))
        out["census"]["built_carry_at_row_steps"] = {"simulation_steps": steps, "carry": at_row}
        out["census"]["equal"]["pool passes"] = cell["passes"] == built_rows["passes"]
        out["census"]["equal"]["pool carry"] = {k: cell["carry"].get(k) for k in at_row} == dict(at_row)

    from mimarsinan.deployment_record.design_point.pricing import executed_record
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
    from mimarsinan.deployment_record.quantities.from_record import from_record
    from mimarsinan.mapping.support.wavefront_stages import input_slot, program_slots
    facts = scheduled.facts
    context = candidate_context_from_platform(dict(platform), latency_steps=facts.latency_steps,
                                              programming=facts.programming, carry=facts.carry, passes=facts.passes)
    cand_q = from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                            context=context)
    record = executed_record(mapping, dict(platform), weight_bits=bits, label="thru:e2 check", timesteps=STEPS,
                             pass_transfer=TRANSFER)
    rec_q = from_record(record)
    slotted = {}
    for key in ("pass_count", "sync_count", "segment_cores", "cores_allocated", "macs", "cells_used",
                "reprogrammed_cores", "reprogrammed_bytes", "reprogram_passes", "carry_out_bytes", "carry_in_bytes",
                "carry_peak_live_bytes", "input_span_syncs", "input_span_cores"):
        c = cand_q.get(key).value if cand_q.has(key) else None
        r = rec_q.get(key).value if rec_q.has(key) else None
        slotted[key] = {"candidate": c, "record": r, "equal": c is not None and r is not None
                        and abs(float(c) - float(r)) <= 1e-9 * max(1.0, abs(float(r)))}
    out["census"]["slotted_per_inference"] = slotted
    out["census"]["equal"]["slotted quantities"] = all(v["equal"] for v in slotted.values())
    out["census"]["program_slots"] = program_slots(mapping.stages)
    out["census"]["stage_slots"] = sorted({input_slot(s).slot for s in neural if input_slot(s) is not None})
    stage("census compared", equal=out["census"]["equal"])


    x = integer_inputs(program, args.inputs, seed=1)
    flat = x.reshape(x.shape[0], -1)
    out["inputs_pairwise_distinct"] = all(not torch.equal(flat[a], flat[b]) for a in range(len(flat))
                                          for b in range(a + 1, len(flat)))
    with torch.no_grad():
        got = ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu")(x.double())
    stage("value run")
    codes, steps = run_golden(mapping, x)
    gold = torch.as_tensor(codes * steps).double().reshape(got.shape)
    stage("int64 golden")
    with torch.no_grad():
        want = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=program.ir), dtype=torch.float64,
                                   device="cpu")(x.double()).reshape(got.shape)
    stage("identity program")
    per_input = [int(((got != want) | (gold != want)).reshape(x.shape[0], -1)[k].sum()) for k in range(x.shape[0])]
    out["exact"] = {"outputs": int(want.numel()), "nonzero": int((want != 0).sum()),
                    "differing_per_input": per_input,
                    "bitwise_equal_value": bool(torch.equal(got, want)
                                                and torch.equal(torch.signbit(got), torch.signbit(want))),
                    "max_abs_delta_value_vs_identity": float((got - want).abs().max()),
                    "max_abs_delta_golden_vs_identity": float((gold - want).abs().max()),
                    "differing_value": int((got != want).sum()), "differing_golden": int((gold != want).sum())}
    stage("exactness", **out["exact"])


    report = check_deployment(mapping, ir_graph=program.ir,
                              declaration=CheckDeclaration(core_types=tuple(dict(c) for c in platform["cores"]),
                                                           **VALUE))
    out["static_check"] = {"theorem1": report.theorem1, "established": report.established("theorem1"),
                           "failures": report.failures_by_condition()}
    stage("static check", theorem1=report.theorem1[:200])
    out["wall_s"] = round(time.time() - start, 1)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
