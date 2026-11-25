""
import json
import math
import sys

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import registry as regmod
from mimarsinan.deployment_record.platform_physics.area_rule import typerule
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as eq
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import decomposition, parity
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    band_factors, converter_area, core_area, held_outside_part, type_gamma, type_value)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

FA = json.load(open(sys.argv[1]))
R = regmod.load_registry()
POINTS = {p["name"]: p for p in R.published_points()}
TERM = "chip_area_mm2"


FORM_I = {
    "jiang2023a macro": [
        {"label": "plain logic-rule 6T (k 1.000)", "print_label": "jiang2023a macro k=1.000",
         "k_plain": True},
        {"label": "analog kinds' mean (k 1.706)", "print_label": "jiang2023a macro k=1.706",
         "cell_kinds": ["cim_amortized_6t_charge_domain", "cim_8t_multiplying_bitcell_mom", "cim_10t1c_mom"]}],
    "IMAGINE": [
        {"label": "Valavi's kind (k 1.800), SAR law 8 b", "print_label": "IMAGINE k=1.800",
         "cell_kinds": ["cim_8t_multiplying_bitcell_mom"], "converter": ["sar_dedicated", "point"]},
        {"label": "CAP-RAM and Valavi kinds (k 1.576), SAR law 8 b", "print_label": "IMAGINE k=1.576",
         "cell_kinds": ["cim_amortized_6t_charge_domain", "cim_8t_multiplying_bitcell_mom"],
         "converter": ["sar_dedicated", "point"]}],
    "CAP-RAM": [
        {"label": "plain logic-rule 6T, SAR law 7 b", "print_label": "CAP-RAM k=1.000, conv SAR law 439",
         "k_plain": True, "converter": ["sar_dedicated", "point"]},
        {"label": "jiang, Valavi, IMAGINE kinds (k 1.533), SAR law 7 b", "print_label": "CAP-RAM k=1.533, conv SAR law 439",
         "cell_kinds": ["cim_logic_rule_6t_digital", "cim_8t_multiplying_bitcell_mom", "cim_10t1c_mom"],
         "converter": ["sar_dedicated", "point"]}],
    "NeuRRAM": [
        {"label": "GS/s SAR mean, 8 b", "print_label": "NeuRRAM conv GS/s SAR mean 1249",
         "converter": ["sar_gsps", 8.0]},
        {"label": "SAR law at 6 b", "print_label": "NeuRRAM conv SAR law 6 b 252", "converter": ["sar_dedicated", "point"]}],
    "HERMES-core (JSSC)": [
        {"label": "HERMES-64's CCO share", "print_label": "HERMES-core (JSSC) conv HERMES-64 ADC share 1529",
         "converter": ["cco_time_based", None]},
        {"label": "SAR law 8 b", "print_label": "HERMES-core (JSSC) conv SAR law 8 b 739", "converter": ["sar_dedicated", 8.0]}],
}


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def factor(r):
    return max(r, 1.0 / r)


def row_dict(r):
    return {"chip": r.chip, "type": r.type, "scope": r.scope, "primary": r.primary, "class": r.cls, "model": r.model,
            "scope_mismatch": r.scope_mismatch, "published_mm2": r.published_mm2, "ratio": r.ratio,
            "estimate_mm2": None if r.ratio is None else r.ratio * r.published_mm2, "refused": r.refused,
            "refused_by": r.refused_by, "array_um2": r.array_um2, "eta": r.eta, "gamma": r.gamma, "count": r.count,
            "core_um2": None if r.terms is None else r.terms.core_um2,
            "converters_um2": None if r.terms is None else r.terms.converters_um2,
            "readings": [{"label": x.label, "converter_um2": x.converter_um2, "ratio": x.ratio,
                          "type_default": x.type_default} for x in r.readings],
            "own_units": list(POINTS[r.chip].get("own_units") or []), "node_nm": POINTS[r.chip]["node_nm"],
            "published_ids": next((s.get("published_ids") for s in POINTS[r.chip]["scopes"]
                                   if s["scope"] == r.scope), None)}


def loihi2_readings(rows):
    ""
    name = "Loihi 2"
    point = POINTS[name]
    view = parity.view(R, point, "held_out")
    primary = next(s for s in point["scopes"] if s["primary"])
    published = R.evaluate(primary["published_mm2"])
    nodes = sorted((n["node_nm"] for n in R.files["node_series"]["nodes"]))
    n1, n2 = nodes[0], nodes[1]
    s1, s2 = R.sigma(n1), R.sigma(n2)
    node = float(point["node_nm"])
    extrapolated = math.exp(math.log(s1) + (math.log(s2) - math.log(s1)) * (math.log(node) - math.log(n1)) /
                            (math.log(n2) - math.log(n1)))
    intel7 = R.evidence.value("S-SRAM-10")
    out = {"published_mm2": published, "node_nm": node, "refusal": next(r["refused"] for r in rows
                                                                          if r["chip"] == name and r["primary"]),
           "readings": []}
    for label, kind, cell in (("Intel 7's HD cell (S-SRAM-10) declared: an upper bound on Intel 4's", "bound: at most",
                               intel7),
                              (f"the 6T series extrapolated below {n1:g} nm (log-log through {n1:g} and {n2:g} nm)",
                               "extrapolation, bounds nothing", extrapolated)):
        declared = eq.Declared(node, None, cell, None, None)
        terms = typerule.core_terms(view, point["type"], declared, parity.counts(R, point), None)
        est, gamma, count = parity.scope_mm2(view, R, point, primary, terms)
        out["readings"].append({"label": label, "kind": kind, "cell_um2": cell, "estimate_mm2": est,
                                "ratio": est / published})
    return out


def form_i_unit(held):
    ""
    per = {}
    for row in held:
        if not (row["primary"] and not row["model"] and not row["scope_mismatch"]):
            continue
        chip = row["chip"]
        if chip in FORM_I:
            point = POINTS[chip]
            view = parity.view(R, point, "strict")
            primary = next(s for s in point["scopes"] if s["primary"])
            published = R.evaluate(primary["published_mm2"])
            counts = parity.counts(R, point)
            declared = parity.declared(point)
            own_bits = R.evaluate(point["converter"]["bits"]) if (point.get("converter") or {}).get("bits") else None
            readings = []
            for rd in FORM_I[chip]:
                if rd.get("k_plain"):
                    k = 1.0
                    cell = view.value("cell_kinds:kinds.sram_6t_logic_rule.factor_of_sigma") * \
                        view.sigma(float(point["node_nm"]))
                elif rd.get("cell_kinds"):
                    k = gm([view.value(f"cell_kinds:kinds.{kind}.factor_of_logic_6t") for kind in rd["cell_kinds"]])
                    cell = k * view.value("cell_kinds:kinds.sram_6t_logic_rule.factor_of_sigma") * \
                        view.sigma(float(point["node_nm"]))
                else:
                    k, cell = None, eq.core_cell_um2(view, point["type"], declared)
                verdict, unit, kind, bits = [], 0.0, None, None
                if rd.get("converter"):
                    kind, bits = rd["converter"]
                    bits = own_bits if bits == "point" else bits
                    unit = converter_area(view, kind, bits, check=False).value
                    declared_kind = (point.get("converter") or {}).get("kind")
                    if kind != declared_kind:
                        verdict.append(f"kind substitution: the point declares {declared_kind}; rule E prices a "
                                       "declared kind from its own units only")
                    try:
                        converter_area(view, kind, bits, check=True)
                    except AreaRuleRefusal as refused:
                        verdict.append(str(refused))
                elif counts.converters:
                    unit = eq.converter_um2(view, point["type"], declared, own_bits)
                terms = core_area(counts.array, cell, eq.type_eta(view, point["type"]),
                                  type_value(view, point["type"], "s"), n_conv=counts.converters,
                                  conv_um2=unit if counts.converters else 0.0)
                est, _g, _c = parity.scope_mm2(view, R, point, primary, terms)
                readings.append({"label": rd["label"], "print_label": rd["print_label"], "k": k, "cell_um2": cell,
                                 "converter": None if kind is None else {"kind": kind, "bits": bits, "unit_um2": unit},
                                 "ratio": est / published, "factor": factor(est / published),
                                 "rule_E": "priced" if not verdict else f"refused ({len(verdict)} reason(s))",
                                 "rule_E_reasons": verdict})
            fs = [x["factor"] for x in readings]
            per[chip] = {"own_unit_row": True, "readings": readings,
                         "whisker": [min(x["ratio"] for x in readings), max(x["ratio"] for x in readings)],
                         "factor_low": min(fs), "factor_high": max(fs)}
        elif row["ratio"] is not None:
            f = factor(row["ratio"])
            per[chip] = {"own_unit_row": False, "held_out": row["ratio"], "factor_low": f, "factor_high": f}
        else:
            per[chip] = {"own_unit_row": False, "enters": "nothing: a named refusal held out, no own unit",
                         "refused": row["refused"]}
    entered = {c: x for c, x in per.items() if "factor_low" in x}
    low = max(entered.items(), key=lambda kv: kv[1]["factor_low"])
    high = max(entered.items(), key=lambda kv: kv[1]["factor_high"])
    return {"per_chip": per, "low": low[1]["factor_low"], "low_chip": low[0], "high": high[1]["factor_high"],
            "high_chip": high[0],
            "readings_total": sum(len(x.get("readings") or []) for x in per.values()),
            "readings_rule_E_refuses": sum(1 for x in per.values() for r in x.get("readings") or []
                                           if r["rule_E_reasons"])}


def stacks_of(held_rows):
    out = {}
    for rule in ("published", "macro"):
        stacks = decomposition.array_decomposition(R, held_rows, rule)
        out[rule] = {"stacks": [s._asdict() for s in stacks], "sram": decomposition.sram_array_fact(stacks)}
    return out


def ladder_prices():
    ""
    p = get_platform_physics(FA["ladder_readings"]["description"])
    out = []
    for member in FA["ladder"]:
        wb = 8.0 if "VAL8" in member["member"] else 4.0
        for row in member["rows"]:
            rows_, cols_ = (int(x) for x in row["geometry"].split("x"))
            n = int(row["cores"])
            vals = {"cores_physical": n, "axons_physical": n * rows_, "neurons_physical": n * cols_,
                    "cells_physical": n * rows_ * cols_, "weight_bits": wb, "tiles": math.ceil(n / 4)}
            q = Quantities({k: QuantityValue(float(x), "static") for k, x in vals.items()})
            pr = price_absolute(q, p, geometry=[CoreGeometry(rows_, cols_, n)])
            term = {t.name: t for t in pr.terms}[TERM]
            flags = {f.kind: f.value for f in pr.flags if f.name == TERM}
            out.append({"member": member["member"], "row": row["row"], "status": row["status"], "cores": n,
                        "geometry": row["geometry"], "weight_bits": wb, "nominal": term.band.nominal,
                        "low": term.band.low, "high": term.band.high, "t1_reading": flags.get("t1_reading"),
                        "fixed_outside_reading": flags.get("fixed_outside_reading"),
                        "per_core": term.band.nominal / n, "per_core_t1": flags.get("t1_reading") / n})
    return out


def main():
    out = {"format": "uarea-w9-unit-facts/1"}
    params = {"eta_T1": "types/t1:eta", "s_T1": "types/t1:s", "gamma_T1": "types/t1:gamma",
              "eta_T2": "types/t2_analog:eta", "s_T2a": "types/t2_analog:s", "s_T2d": "types/t2_digital:s",
              "s_T3": "types/t3:s", "s_T4": "types/t4:s", "gamma_T4": "types/t4:gamma",
              "kL": "cell_kinds:kinds.sram_6t_logic_rule.factor_of_sigma", "outside_part_held": "types/t1:outside_parts"}
    out["params"] = {k: R.value(v) for k, v in params.items()}
    out["params"]["sigma"] = {str(n): R.sigma(n) for n in (65, 45, 32, 28, 22, 16, 14, 10, 7, 5)}
    out["params"]["gamma_T2d"] = type_gamma(R, "T2_digital")
    out["params"]["bands"] = {t: list(band_factors(R, t)) for t in ("T1", "T2_digital", "T2_analog", "T3", "T4")}
    held_part = held_outside_part(R, "T1")
    out["params"]["outside_parts"] = {"held": held_part.value, "parts": dict(held_part.parts)}
    share = R.files[R.type_file("T1")]["s"]
    value, kept, dropped = R.fitted(share, (), "types/t1:s")
    pts = []
    for point in share["recompute"]["points"]:
        try:
            pts.append({"label": point["label"], "value": R.evaluate(point["expression"])})
        except AreaRuleRefusal as refused:
            pts.append({"label": point["label"], "dropped": str(refused)})
    out["share_fit"] = {"value": value, "kept": list(kept), "dropped": list(dropped), "points": pts}
    rows = {mode: [row_dict(r) for r in parity.estimate(R, mode)] for mode in parity.MODES}
    out["rows"] = rows
    held_objs = parity.estimate(R, "held_out")
    head = parity.headline(held_objs)
    priced = [r for r in head if r.refused is None]
    out["stats"] = parity.stats([r.ratio for r in priced])
    out["stats"]["n_refused"] = sum(1 for r in head if r.refused is not None)
    out["refused"] = [{"chip": r.chip, "type": r.type, "scope": r.scope, "published_mm2": r.published_mm2,
                       "refused": r.refused, "refused_by": r.refused_by,
                       "readings": [{"label": x.label, "ratio": x.ratio, "converter_um2": x.converter_um2,
                                     "type_default": x.type_default} for x in r.readings]}
                      for r in head if r.refused is not None]
    out["stats"]["types_priced"] = sorted({r.type for r in priced})
    out["stats"]["types"] = len({r.type.split("_")[0] for r in priced})
    own = [r for r in head if POINTS[r.chip].get("own_units")]
    out["stats"]["own_unit"] = len(own)
    out["stats"]["own_unit_priced"] = sum(1 for r in own if r.refused is None)
    indep = [r for r in head if r.cls == "independent"]
    ind_priced = [r.ratio for r in indep if r.refused is None]
    out["stats"]["independent_rows"] = [{"chip": r.chip, "ratio": r.ratio, "refused": r.refused} for r in indep]
    out["stats"]["independent"] = len(ind_priced)
    out["stats"]["independent_min"] = min(ind_priced)
    out["stats"]["independent_max"] = max(ind_priced)
    counts = {}
    for r in priced:
        counts[r.type] = counts.get(r.type, 0) + 1
    out["stats"]["priced_per_type"] = counts
    out["strict_form"] = parity.strict_form(parity.estimate(R, "strict"))
    out["strict_spread"] = parity.strict_spread(parity.estimate(R, "strict"))
    out["strict_spread_all_readings"] = parity.strict_spread(parity.estimate(R, "strict"), "all")
    out["loihi2_readings"] = loihi2_readings(rows["held_out"])
    out["form_i_unit"] = form_i_unit(rows["held_out"])
    out["stacks"] = stacks_of(held_objs)
    out["ladder"] = ladder_prices()
    json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
    s = out["stats"]
    print(json.dumps({"n": s["n"], "n_refused": s["n_refused"], "geomean": s["geomean"], "worst": s["worst_factor"],
                      "within_2x": s["within_2x"], "types": s["types"], "own_unit": s["own_unit"],
                      "own_unit_priced": s["own_unit_priced"], "independent": s["independent"],
                      "indep": [s["independent_min"], s["independent_max"]],
                      "strict": out["strict_form"], "loihi2": [x["ratio"] for x in out["loihi2_readings"]["readings"]],
                      "form_i": [out["form_i_unit"]["low"], out["form_i_unit"]["high"]],
                      "sram": out["stacks"]["published"]["sram"]["max_deviation"],
                      "sram_macro_mutant": out["stacks"]["macro"]["sram"]["max_deviation"],
                      "share": out["share_fit"]["value"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
