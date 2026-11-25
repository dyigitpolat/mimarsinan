""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_gate_checkpoint as ig

N = 5000


PROBE = {"softmax.O8": (3624, 3638), "softmax.O4": (3615, 3643), "softmax.T": (3634, 3642),
         "relu.O8": (3582, 3587), "relu.T": (3606, 3616)}
FLOAT_JOB = {"softmax": ("F", 3672), "relu": ("F-R", 3647)}


def paired(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> dict:
    pa, pb = a == labels, b == labels
    only_a, only_b = int((pa & ~pb).sum()), int((~pa & pb).sum())
    n = len(labels)
    se = 100 * ((only_a + only_b - (only_a - only_b) ** 2 / n) ** 0.5) / n
    return {"delta_pp": round(100 * (int(pa.sum()) - int(pb.sum())) / n, 2), "a_only": only_a, "b_only": only_b,
            "paired_se_pp": round(se, 3)}


def gate_count(key: str) -> int:
    job, read = ig.GATE_RECORD[key]
    return int(ig.gate_record(job)["counts"][read])


def gate_predictions(key: str):
    job, read = ig.GATE_RECORD[key]
    name = "pred_qat.npy" if job.startswith("QF-") else f"pred_{read}.npy"
    return np.load(ig.GATE / "runs" / job / name).astype(np.int64)


CENSUS_FORMS = {"per_kind": frozenset({"payload_codes", "operand_core_outputs", "host_op_outputs", "outputs"}),
                "flat": frozenset({"payload_codes", "scores", "softmax_outputs", "outputs"})}

CENSUS_NAMES = {"operand_core_outputs": ("operand_core_outputs", "scores"), "outputs": ("outputs",),
                "payload_codes": ("payload_codes",)}


class CensusRefused(Exception):
    ""

    def __init__(self, name: str, detail: str):
        super().__init__(f"{name}: {detail}")
        self.name, self.detail = name, detail


def census_form(census: dict) -> str:
    ""
    names = frozenset(census)
    for form, have in CENSUS_FORMS.items():
        if names == have:
            return form
    return "names: " + ", ".join(sorted(names))


def census_count(census: dict, name: str):
    ""
    for alias in CENSUS_NAMES.get(name, (name,)):
        if alias in census:
            return census[alias]
    return None


def host_op_outputs(census: dict) -> dict:
    ""
    if "host_op_outputs" in census:
        return dict(census["host_op_outputs"])
    return {"softmax": census["softmax_outputs"]} if "softmax_outputs" in census else {}


def _is_count(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def census_sum(censuses) -> dict:
    ""
    def add(acc: dict, c: dict, where: str) -> None:
        for k, v in c.items():
            if isinstance(v, dict):
                if k in acc and not isinstance(acc[k], dict):
                    raise CensusRefused("CENSUS_VALUE", f"{where}{k} is a count in one census, a mapping in another")
                add(acc.setdefault(k, {}), v, f"{where}{k}.")
            elif _is_count(v):
                if isinstance(acc.get(k), dict):
                    raise CensusRefused("CENSUS_VALUE", f"{where}{k} is a mapping in one census, a count in another")
                acc[k] = int(acc.get(k, 0)) + int(v)
            else:
                raise CensusRefused("CENSUS_VALUE", f"{where}{k} is {v!r}, neither a count nor a mapping of counts")

    out: dict = {}
    for c in censuses:
        add(out, c, "")
    return out


def certificate_of(shards: list) -> dict:
    if not shards:
        return {"batches": 0}
    edges: dict = {}
    payload = {"cells_compared": 0, "mismatched": 0, "nonzero_payload_codes": 0}
    rows = [row for s in shards for row in s["rows"]]
    forms = sorted({census_form(row["nonzero_census"]) for row in rows})
    if len(forms) > 1:
        raise CensusRefused("CENSUS_FORMS_DIFFER", f"the shards of one deployment write the census in {len(forms)} "
                                                   f"forms ({'; '.join(forms)}): their counts are not summed together")
    for row in rows:
        for e in row["edges"]:
            acc = edges.setdefault(e["name"], {"compared": 0, "mismatched": 0, "max_abs_delta": 0.0})
            acc["compared"] += e["compared"]
            acc["mismatched"] += e["mismatched"]
            acc["max_abs_delta"] = max(acc["max_abs_delta"], e["max_abs_delta"])
        for k in payload:
            payload[k] += row["payload"][k]
    census = census_sum(row["nonzero_census"] for row in rows)
    batches = sorted(row["batch"] for row in rows)
    required = ("R (NF/SCM)", "golden/SCM", "golden/HCM", "C (SCM/HCM windows)", "host ops (bitwise)")


    applicable = (all(acc["mismatched"] == 0 for acc in edges.values())
                  and all(edges.get(name, {}).get("compared", 0) > 0 for name in required)
                  and payload["mismatched"] == 0 and (census_count(census, "outputs") or 0) > 0)
    return {"batches": len(batches), "images": int(sum(row["images"] for row in rows)),
            "batch_ranges": [s["batches"] for s in shards], "every_batch_passed": all(r["passed"] for r in rows),
            "applicable_edges_exact": bool(applicable),
            "edges": edges, "payload": payload, "nonzero_census": census, "census_form": forms[0],
            "scm_hcm_argmax_equal_every_batch": all(r["scm_hcm_argmax_equal"] for r in rows),
            "wall_s_per_batch_mean": round(float(np.mean([r["wall_s"] for r in rows])), 1),
            "_shards": [(s["batches"][0] * s["batch"], np.asarray(s["pred_hcm"])) for s in shards]}


def main() -> int:
    build, out = Path(sys.argv[1]), Path(sys.argv[2])
    labels = ig.load_gate_data().labels
    deployments, preds = {}, {}
    for key in ig.JOBS:
        counts = sorted(glob.glob(str(build / "acceptance3" / "counts" / f"{key}.count.*.json")))
        shards = [json.loads(Path(p).read_text()) for p in
                  sorted(glob.glob(str(build / "acceptance3" / "certs*" / f"{key}.certify.*.json")))]
        row: dict = {"job": ig.JOBS[key]}
        if counts:
            c = json.loads(Path(counts[-1]).read_text())
            pred = np.asarray(c["pred_hcm"])
            if len(pred) != N:
                raise SystemExit(f"REFUSED: {key} count run read {len(pred)} images, not {N}")
            preds[key] = pred
            row.update(deployed_correct=int((pred == labels).sum()), nf_correct=c["correct_nf"],
                       nf_hcm_predictions_equal=bool(np.array_equal(pred, np.asarray(c["pred_nf"]))),
                       images_with_any_logit_differing_hcm_nf=c["images_with_any_logit_differing_hcm_nf"],
                       ir=c["ir"], lowered=len(c["lowered"]), packing=c["packing"],
                       import_steps=c["provenance"].get("gate_steps_installed"),
                       import_worst_relative_step_error=c["provenance"].get("import_worst_relative_step_error"),
                       calibrated_ranges=len(c["provenance"].get("calibrated", [])),
                       stage_a_record_check=c["provenance"].get("stage_a_record_check"),
                       checkpoint_sha256=c["provenance"]["sha256"], split_sha256=c["split_sha256"],
                       calibration_sha256=c["calibration_sha256"], wall_s=c["wall_s"])
        row["gate_record"] = gate_count(key)
        if key in PROBE:
            row["probe_estimate"], row["gate_cpu_reread"] = PROBE[key]
        if key in preds:
            gate = gate_predictions(key)
            row["agreement_with_gate_predictions"] = round(float((gate == preds[key]).mean()), 4)
            row["paired_vs_gate_record"] = paired(preds[key], gate, labels)
        try:
            cert = certificate_of(shards)
        except CensusRefused as refused:
            print(f"REFUSED {key}: {refused}", file=sys.stderr, flush=True)
            return 2
        if cert.get("batches"):
            parts = cert.pop("_shards")
            if key in preds:
                cert["hcm_predictions_equal_count_run"] = all(
                    np.array_equal(pred, preds[key][first:first + len(pred)]) for first, pred in parts)
                cert["certified_images_correct"] = int(sum((pred == labels[first:first + len(pred)]).sum()
                                                           for first, pred in parts))
        row["certificate"] = cert
        deployments[key] = row
    pairs = {}
    for form in ("softmax", "relu"):
        twin = f"{form}.T"
        f_job, f_count = FLOAT_JOB[form]
        for bits in (8, 6, 4):
            key = f"{form}.O{bits}"
            if key not in preds or twin not in preds:
                continue
            p = paired(preds[key], preds[twin], labels)
            twin_count = deployments[twin]["deployed_correct"]
            valid = twin_count >= f_count - 50
            cls = ("not determinable (degraded twin)" if not valid else
                   "holds at one seed" if p["delta_pp"] >= -1.0 else
                   "fails at one seed" if p["delta_pp"] < -2.5 else "inconclusive at one seed")
            pairs[f"{form}.{bits}"] = {**p, "operand": deployments[key]["deployed_correct"], "twin": twin_count,
                                      "float_job": f_job, "float_record": f_count,
                                      "twin_minus_float_pp": round(100 * (twin_count - f_count) / N, 2),
                                      "twin_valid": valid, "class": cls,
                                      "gate_record_delta_pp": round(100 * (deployments[key]["gate_record"]
                                                                           - deployments[twin]["gate_record"]) / N, 2)}
    stop = pairs.get("softmax.8")
    result = {"n_images": N, "deployments": deployments, "pairs": pairs,
              "stop_line": None if stop is None else {"delta_pp": stop["delta_pp"], "below_minus_1pp":
                                                       stop["delta_pp"] < -1.0},
              "script": {"name": "aggregate_gate_r2.py", "revision": 2}}
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"pairs": pairs, "stop_line": result["stop_line"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
