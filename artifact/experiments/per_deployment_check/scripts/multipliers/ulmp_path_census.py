""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from mimarsinan.mapping.export.law_rules import adc_step_exponent, multiplier_and_shift, round_ints
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import (
    accumulator_bounds,
    accumulator_width,
    clamp_reach,
    grid_levels,
)

try:
    from mimarsinan.mapping.lanes.law.multiplier_paths import refuse_lost_paths
except ImportError:
    refuse_lost_paths = None

W34_ADC_BITS = (4, 5, 6, 8)


def own_shift(ratio, mult_bits, core):
    try:
        return multiplier_and_shift(float(ratio), mult_bits, core=core)[1]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def producer_codes(node, row, nodes):
    ""
    source = node.input_sources.flatten()[int(row)]
    producer = nodes.get(int(source.node_id)) if source.node_id >= 0 else None
    law = getattr(producer, "readout_law", None) if isinstance(producer, NeuralCore) else None
    if law is None:
        return None
    j = int(source.index)
    return int(np.asarray(law.lo).reshape(-1)[j]), int(np.asarray(law.hi).reshape(-1)[j])


def input_levels(node, nodes, data_rows):
    ""
    seen = [producer_codes(node, r, nodes) for r in data_rows]
    seen = [c for c in seen if c is not None]
    if seen:
        return max(max(abs(a), abs(b)) for a, b in seen)
    grid = getattr(node, "boundary_grid", None)
    if grid is not None and getattr(grid, "armed", False):
        low, high = grid.code_range if hasattr(grid, "code_range") else (-grid.levels, grid.levels)
        return max(abs(int(low)), abs(int(high)))
    return None


def law_paths(node, law, mult_bits, nodes, ir):
    f = lambda a: np.asarray(a).reshape(-1)
    theta = float(node.threshold)
    in_step, out_step = float(f(law.input_step)[0]), float(f(law.output_step)[0])
    m0s, shifts = f(law.m0).astype(np.int64), f(law.shift).astype(np.int64)
    rows = f(getattr(law, "operand_row", np.full(m0s.size, -1))).astype(np.int64)
    opm = f(getattr(law, "operand_m", np.zeros(m0s.size))).astype(np.int64)
    opstep = f(getattr(law, "operand_step", np.zeros(m0s.size))).astype(np.float64)
    columns = int(m0s.size)
    operand_cols = np.flatnonzero(rows >= 0)
    rec = {"core": node.name, "perceptron": node.perceptron_index, "lane_owner": getattr(node, "lane_owner", None),
           "columns": columns, "operand_columns": int(operand_cols.size), "theta": theta,
           "input_offset": float(f(getattr(law, "input_offset", [0.0]))[0])}
    if len(set(m0s.tolist())) != 1 or len(set(shifts.tolist())) != 1:
        rec["note"] = "per-column multipliers or shifts (a grouped law): read at its first column"
    r0 = in_step / (theta * out_step)
    m0, s = int(m0s[0]), int(shifts[0])
    rec["paths"] = 1 + (1 if operand_cols.size else 0)
    rec["shift"] = s
    rec["data"] = {"ratio": r0, "M": m0, "bits": int(abs(m0)).bit_length(), "own_shift": own_shift(r0, mult_bits,
                                                                                                     node.name)}
    ratio1 = [None] * columns
    for c in operand_cols.tolist():
        ratio1[c] = float(opstep[c] / out_step)
    distinct = sorted(set(r for r in ratio1 if r is not None))
    rec["operand"] = []
    for r1 in distinct:
        cols = [c for c in operand_cols.tolist() if ratio1[c] == r1]
        ms = sorted(set(int(opm[c]) for c in cols))
        rec["operand"].append({"ratio": r1, "own_shift": own_shift(r1, mult_bits, node.name), "M": ms,
                               "bits": [int(abs(m)).bit_length() for m in ms], "columns": len(cols)})
    owns = [rec["data"]["own_shift"]] + [e["own_shift"] for e in rec["operand"]]
    if all(isinstance(v, int) for v in owns):
        shared = min(owns)
        rec["recomputed_equal"] = bool(shared == s and int(round_ints(np.ldexp(r0, shared))) == m0 and all(
            len(e["M"]) == 1 and int(round_ints(np.ldexp(e["ratio"], shared))) == e["M"][0] for e in rec["operand"]))
        rec["binding"] = "data" if owns[0] == shared else "operand"
        rec["data"]["bits_lost_to_sharing"] = int(owns[0] - shared)
        for e in rec["operand"]:
            e["bits_lost_to_sharing"] = int(e["own_shift"] - shared)
    if rec["paths"] == 1:
        return rec

    matrix = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
    sources = node.input_sources.flatten()
    on = np.array([src.is_always_on() for src in sources], dtype=bool)
    data = [i for i, src in enumerate(sources) if not (src.is_off() or src.is_always_on()) and i not in set(
        rows[rows >= 0].tolist())]
    levels = input_levels(node, nodes, data)
    rec["input_levels"] = levels
    bounds = accumulator_bounds(matrix[~on, :], levels) if levels is not None else None
    bias = f(law.bias).astype(np.int64)
    lo, hi = int(f(law.lo)[0]), int(f(law.hi)[0])
    adc_shift = f(law.adc_shift).astype(np.int64)
    codes = [producer_codes(node, int(rows[c]), nodes) if rows[c] >= 0 else None for c in range(columns)]
    rec["output_codes"] = [lo, hi]
    rec["operand_codes"] = sorted(set(tuple(cd) for cd in codes if cd is not None))
    floors = {"data_lost": m0 == 0, "operand_lost": any(m == 0 for e in rec["operand"] for m in e["M"])}
    if m0 != 0 and bounds is not None:
        terms = [(0, 0) if cd is None else (min(int(opm[c]) * cd[0], int(opm[c]) * cd[1]),
                                            max(int(opm[c]) * cd[0], int(opm[c]) * cd[1]))
                 for c, cd in enumerate(codes)]
        reach = [min(int(b), clamp_reach(m0, int(bj), s, lo, hi, t)) for b, bj, t in zip(bounds, bias.tolist(), terms)]
        x0 = max(r + ((1 << int(t)) >> 1) for r, t in zip(reach, adc_shift.tolist()))
        starved = isinstance(rec["data"]["own_shift"], int) and rec["data"]["own_shift"] > s
        floors.update(data_starved=starved, data_reach=x0, data_reach_over_2s=x0 / (1 << s),
                      data_floor_fails=bool(starved and x0 > (1 << s)))
        width = accumulator_width(max(bounds, default=0))
        hypo = {}
        for bits in W34_ADC_BITS:
            if bits >= width:
                xb = max(reach)
            else:
                lv = grid_levels(bits)
                xb = max(r + ((1 << adc_step_exponent(r, lv, core=node.name)) >> 1) for r in reach)
            hypo[str(bits)] = {"reach_over_2s": xb / (1 << s), "fails": bool(starved and xb > (1 << s))}
        floors["data_floor_at_adc_bits"] = hypo
    op_fail = False
    for e in rec["operand"]:
        x1 = max((abs(v) for c in range(columns) if ratio1[c] == e["ratio"] and codes[c] for v in codes[c]), default=0)
        e["reach"] = x1
        e["floor_fails"] = bool(e.get("bits_lost_to_sharing", 0) > 0 and x1 > (1 << s))
        op_fail = op_fail or e["floor_fails"]
    floors["operand_floor_fails"] = op_fail
    rec["floors"] = floors
    if refuse_lost_paths is not None and bounds is not None:
        try:
            refuse_lost_paths(node, shift=s, mult_bits=mult_bits, ratio0=r0, m0=m0, ratio1=ratio1, m1=opm.tolist(),
                              bias=bias.tolist(), acc_bounds=bounds, adc_shift=adc_shift.tolist(), lo=lo, hi=hi,
                              codes=codes)
            rec["tree_rule"] = "kept"
        except Exception as exc:
            rec["tree_rule"] = f"refused: {type(exc).__name__}: {str(exc)[:300]}"
    return rec


def ir_of(path: Path):
    if not path.exists():
        return None, None
    if path.is_dir():
        hits = [h for h in sorted(path.rglob("Soft Core Mapping.ir_graph.pickle")) if "__untrained__" not in str(h)]
        if not hits:
            return None, None
        path = hits[0]
    with open(path, "rb") as fh:
        return pickle.load(fh), path


def census(ir, mult_bits):
    seen, cores_per_law = {}, {}
    cores = 0
    for node in ir.nodes:
        law = getattr(node, "readout_law", None) if isinstance(node, NeuralCore) else None
        if law is None:
            continue
        cores += 1
        cores_per_law[id(law)] = cores_per_law.get(id(law), 0) + 1
        seen.setdefault(id(law), node)
    nodes = {int(n.id): n for n in ir.nodes}
    multi, single_bits = [], {}
    for key, node in seen.items():
        rec = law_paths(node, node.readout_law, mult_bits, nodes, ir)
        rec["cores"] = cores_per_law[key]
        if rec["paths"] == 1:
            single_bits[rec["data"]["bits"]] = single_bits.get(rec["data"]["bits"], 0) + 1
        else:
            multi.append(rec)
    fl = [r["floors"] for r in multi]
    return {"law_cores": cores, "law_objects": len(seen), "multi_path_law_objects": len(multi),
            "multi_path_cores": sum(r["cores"] for r in multi),
            "single_path_law_objects_by_bits": {str(k): v for k, v in sorted(single_bits.items())},
            "multi_path_data_bits": sorted(set(r["data"]["bits"] for r in multi)),
            "multi_path_operand_bits": sorted(set(b for r in multi for e in r["operand"] for b in e["bits"])),
            "data_bits_lost_to_sharing": sorted(set(r["data"].get("bits_lost_to_sharing", -1) for r in multi)),
            "data_lost": sum(1 for x in fl if x["data_lost"]), "operand_lost": sum(1 for x in fl if x["operand_lost"]),
            "data_floor_fails": sum(1 for x in fl if x.get("data_floor_fails")),
            "operand_floor_fails": sum(1 for x in fl if x["operand_floor_fails"]),
            "data_reach_over_2s_max": max((x.get("data_reach_over_2s", 0.0) for x in fl), default=None),
            "w34_adc_floor_fails": {b: sum(1 for x in fl if x.get("data_floor_at_adc_bits", {}).get(b, {}).get(
                "fails")) for b in map(str, W34_ADC_BITS)},
            "w34_adc_reach_over_2s_max": {b: max((x.get("data_floor_at_adc_bits", {}).get(b, {}).get(
                "reach_over_2s", 0.0) for x in fl), default=None) for b in map(str, W34_ADC_BITS)},
            "recomputed_unequal": sum(1 for r in multi if r.get("recomputed_equal") is False),
            "binding": {b: sum(1 for r in multi if r.get("binding") == b) for b in ("data", "operand")},
            "tree_rule_refused": sum(1 for r in multi if str(r.get("tree_rule", "")).startswith("refused")),
            "tree_rule_read": sum(1 for r in multi if "tree_rule" in r),
            "input_offsets": sorted(set(r["input_offset"] for r in multi)),
            "laws": multi}


def main():
    out = Path(sys.argv[1])
    for arg in sys.argv[2:]:
        label, mult, src = arg.split("=", 2)
        t0 = time.time()
        ir, path = ir_of(Path(src))
        head = {"label": label, "source": str(path or src), "mult_bits": int(mult),
                "tree_has_rule": refuse_lost_paths is not None}
        rec = {**head, "status": "no Soft Core Mapping IR on disk"} if ir is None else {**head, **census(ir, int(mult))}
        rec["wall_s"] = round(time.time() - t0, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "laws"}, default=str), flush=True)


if __name__ == "__main__":
    main()
