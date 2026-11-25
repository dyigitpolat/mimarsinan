""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import resource
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
HARNESS = Path("records/10_wave1/cl_cov/harness")
sys.path.insert(0, str(HARNESS))

import torch

import cov_members as CM
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.mappers.operand import operand_mappers
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.module_macs import module_macs
from mimarsinan.mapping.support.operand_lowering import lower_contractions
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.torch_mapping.converter import probe_witness_conversion, trace_and_convert, witness_converter
from mimarsinan.torch_mapping.value_parity import NATIVE_EVAL_IDENTITY, fp64_snapshot
from mimarsinan.torch_mapping.value_parity.comparison import relative_error
from mimarsinan.torch_mapping.value_parity.probe import prepare_witness


PLATFORM = {"cores": [{"max_axons": 768, "max_neurons": 768, "count": 10 ** 6},
                      {"max_axons": 256, "max_neurons": 256, "count": 10 ** 6,
                       "operand": {"payload_bits": 8, "write_source": "host"}}],
            "activation_bits": 8, "readout_law": {"adc_bits": 40, "range": "calibrated", "mult_bits": 16},
            "readout_ranges": "per_consumer_group"}
TV_SIDE = 224


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def shape_of(member: dict, side) -> tuple:
    declared = member["declared_input"]
    if declared["kind"] == "fixed":
        return tuple(declared["shape"])
    s = int(side or TV_SIDE)
    return (int(declared["channels"]), s, s)


def _batched(shape):
    return (1, *tuple(shape)) if shape is not None else None


def host_kinds(ir) -> dict:
    kinds: dict = {}
    for op in ir.get_compute_ops():
        module = op.params.get("module")
        unit = getattr(module, "perceptron", None) or module
        shapes = [_batched(s) for s in op.params.get("input_shapes") or ()]
        macs = 0 if module is None else int(module_macs(unit, _batched(op.input_shape), _batched(op.output_shape),
                                                        shapes))
        key = type(unit).__name__ if module is not None else f"{op.op_type} (no module)"
        row = kinds.setdefault(key, {"ops": 0, "macs": 0})
        row["ops"] += 1
        row["macs"] += macs
    return kinds


def refusal(exc: BaseException) -> dict:
    import traceback
    return {"refused": type(exc).__name__, "message": str(exc).splitlines()[0][:400] if str(exc) else "",
            "where": traceback.format_exc().splitlines()[-6:]}


def agreement(member: dict, model, shape: tuple) -> dict:
    out: dict = {}
    decompose = bool(member["decompose_leaf_modules"])
    t0 = time.time()
    try:
        results = quiet(probe_witness_conversion, copy.deepcopy(model), shape, packaging=MVM_PACKAGING,
                        decompose_leaf_modules=decompose)
        out["verdict"] = [{"stage": r.stage, "ok": bool(r.ok), "relative_error": r.relative_error, "bound": r.bound}
                          for r in results]
    except Exception as exc:
        out["verdict"] = refusal(exc)
        return out
    out["verdict_s"] = round(time.time() - t0, 1)
    gc.collect()
    convert = witness_converter(shape, encoding_layer_placement="subsume", packaging=MVM_PACKAGING,
                                decompose_leaf_modules=decompose)
    with torch.random.fork_rng(devices=[]):
        witness = quiet(prepare_witness, fp64_snapshot(copy.deepcopy(model)), convert, shape,
                        contract=NATIVE_EVAL_IDENTITY, seed=0, batch=6)

    flow = witness.conversion.flow.double().eval()
    batches = (witness.inputs, witness.escalated_inputs, witness.intermediate_inputs)
    with torch.no_grad():
        before = [flow(x) for x in batches]
        want = [witness.reference(x) for x in batches]
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    with torch.no_grad():
        after = [flow(x) for x in batches]
    out["lowered_vs_reference"] = max(relative_error(a, w).relative_error for a, w in zip(after, want))
    out["unlowered_vs_reference"] = max(relative_error(b, w).relative_error for b, w in zip(before, want))
    out["lowered_vs_unlowered"] = max(relative_error(a, b).relative_error for a, b in zip(after, before))
    out["bound"] = float(witness.bound)
    out["lowered_within_bound"] = bool(out["lowered_vs_reference"] <= out["bound"])
    out["witness_lowered"] = len(report.lowered)
    return out


def mac_census(member: dict, model, shape: tuple) -> dict:
    flow = quiet(lambda: trace_and_convert(copy.deepcopy(model), shape, "cpu", packaging=MVM_PACKAGING,
                                           decompose_leaf_modules=bool(member["decompose_leaf_modules"])).flow)
    repr_ = flow.get_mapper_repr()
    try:
        ir = quiet(map_model_to_ir, repr_)
    finally:
        repr_.clear_ir_caches()
    estimate = estimate_onchip_fraction(flow, shape, int(member["num_classes"]), metric="macs")
    host_before = int(count_host_ops(ir))
    kinds_before = host_kinds(ir)
    del ir
    gc.collect()
    report = lower_contractions(repr_, PLATFORM)
    lowered_macs = 0
    for mapper in operand_mappers(repr_):
        form = mapper.lowering.form
        lowered_macs += int(form.banks) * int(form.m) * int(form.k) * int(form.n)
    repr_.assign_perceptron_indices()
    try:
        ir = quiet(map_model_to_ir, repr_)
    finally:
        repr_.clear_ir_caches()
    host_after = int(count_host_ops(ir))
    kinds_after = host_kinds(ir)
    total = int(estimate.total)
    banks = len(getattr(ir, "dynamic_bank_bindings", {}) or {})
    cells = int(sum(b.cells for b in (getattr(ir, "dynamic_bank_bindings", {}) or {}).values()))
    return {"total_macs": total, "host_macs_before": host_before, "estimate_host_macs_before": int(estimate.host),
            "counters_agree_before": host_before == int(estimate.host),
            "host_mac_share_before": host_before / total if total else None,
            "host_macs_after": host_after, "host_mac_share_after": host_after / total if total else None,
            "lowered_product_macs": lowered_macs,
            "host_fall_equals_lowered_macs": host_before - host_after == lowered_macs,
            "lowered": [vars(p) for p in report.lowered], "kept_on_host": [vars(p) for p in report.host],
            "banks": banks, "payload_cells_per_inference": cells,
            "host_ops_by_kind_before": kinds_before, "host_ops_by_kind_after": kinds_after,
            "host_ops_before": sum(r["ops"] for r in kinds_before.values()),
            "host_ops_after": sum(r["ops"] for r in kinds_after.values())}


def classify(row: dict) -> str:
    verdict = row.get("agreement", {}).get("verdict")
    if isinstance(verdict, dict) and "refused" in verdict:
        return f"rejected by name ({verdict['refused']})"
    census = row.get("macs") or {}
    if "refused" in census:
        return f"rejected by name ({census['refused']})"
    if not all(r["ok"] for r in verdict) or not row["agreement"].get("lowered_within_bound"):
        return "value mismatch"
    return "convert, all MACs on cores" if census.get("host_macs_after") == 0 else "convert, CPU share of MACs"


def main() -> int:
    out_dir, member_id = Path(sys.argv[1]), sys.argv[2]
    side = sys.argv[3] if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    members = CM.by_id(CM.census_members())
    if member_id not in members and member_id.endswith(CM.DECOMPOSED):
        member = dict(members[member_id[:-len(CM.DECOMPOSED)]], id=member_id, decompose_leaf_modules=True)
    else:
        member = members[member_id]
    shape = shape_of(member, side)
    t0 = time.time()
    model = CM.build_member(member)
    row = {"member": member_id, "decompose_leaf_modules": member["decompose_leaf_modules"], "input_shape": shape,
           "platform": PLATFORM, "torch": torch.__version__}
    torch.set_num_threads(1)
    row["agreement"] = agreement(member, model, shape)
    if not ("refused" in (row["agreement"].get("verdict") or {}) if isinstance(row["agreement"].get("verdict"), dict)
            else False):
        gc.collect()
        try:
            row["macs"] = mac_census(member, model, shape)
        except Exception as exc:
            row["macs"] = refusal(exc)
    row["class"] = classify(row)
    row["wall_s"] = round(time.time() - t0, 1)
    row["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
    name = member_id.replace("/", "__").replace("[", "_").replace("]", "") + (f"_{side}" if side else "")
    (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("member", "class", "wall_s", "peak_rss_gib")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
