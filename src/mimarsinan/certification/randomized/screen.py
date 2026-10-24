"""The X3a screen report: outcomes per arm, Clopper-Pearson bounds, teeth, coverage and walls."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scipy.stats import beta

from mimarsinan.certification.randomized.arms import ARMS, ON, ON_LONG_WINDOW, ON_SCHEDULED, ON_VALUES
from mimarsinan.certification.randomized.cost_draws import PRESERVED_BY_CONSTRUCTION, partition
from mimarsinan.certification.randomized.generator import DEFAULT_SPACE, MOTIFS
from mimarsinan.certification.randomized.network import CLAIMED
from mimarsinan.certification.randomized.runner import COUNTEREXAMPLE, EQUAL, OUTCOMES, REFUSED, REFUSED_UNNAMED
from mimarsinan.common.refusal import ToolchainRefusal

ON_ARMS = (ON, ON_SCHEDULED, ON_VALUES)
#: The hypothesis-on strata that run the toolchain as shipped (no harness setting): the zero line's pool.
SHIPPED_ON_ARMS = tuple(a for a in ON_ARMS if not ARMS[a].harness_setting)
STRUCTURE = ("multipass", "carried", "coresident", "coalesced", "split", "sign_split")
#: Admitted outcomes: the deployment was built and run, so it either matched or it is a counterexample.
ADMITTED = (EQUAL, COUNTEREXAMPLE, "vacuous")
#: Function-bearing axes the code level holds fixed (every instance; the box prints them as its limit).
FIXED_AXES = {
    "comparator": "'<=' (fire at V >= theta)",
    "reset": "subtractive (firing mode Default)",
    "membrane": "unbounded (float64 executors)",
    "encoder": "Uniform comb, no phase dither (deterministic); the R4 arm alone declares the phase-dithered "
               "Uniform encode, on both programs",
    "granularity": "per-cycle cores (per-event cores are the FPGA branch's)",
    "value_law": "G2's integer readout law (E283, from CL1): a 16-bit activation grid at step 1 (codes +-32,767), the "
                 "identity ADC (32 bits), a 16-bit multiplier; every value run also against the int64 golden at atol 0",
}


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """The two-sided exact interval for ``k`` events in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def one_sided_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """The one-sided upper Clopper-Pearson bound (``1 - 0.05^(1/n)`` at ``k = 0``)."""
    if n <= 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def arm_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    outcomes = Counter(r["outcome"] for r in rows)
    admitted = [r for r in rows if r["outcome"] in ADMITTED]
    counterexamples = [r for r in admitted if r["outcome"] == COUNTEREXAMPLE]
    walls = [float(r["wall_s"]) for r in admitted]
    out: Dict[str, Any] = {
        "instances": len(rows),
        "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES},
        "admitted": len(admitted),
        "counterexamples": len(counterexamples),
        "rate_interval_95": clopper_pearson(len(counterexamples), len(admitted)),
        "upper_bound_95": one_sided_upper(len(counterexamples), len(admitted)),
        "median_wall_s": statistics.median(walls) if walls else None,
        "exercised": {k: sum(1 for r in admitted if r.get(k)) for k in STRUCTURE},
        "semantics": dict(Counter(r["semantics"] for r in admitted)),
        "classes": dict(Counter(tag for r in counterexamples for tag in r.get("classes", ()))),
        "unnamed_refusals": dict(Counter(_kind(r["detail"]) for r in rows if r["outcome"] == REFUSED_UNNAMED)),
        "named_refusals": dict(Counter(_kind(r["detail"]) for r in rows if r["outcome"] == REFUSED)),
        "errors": dict(Counter(_kind(r["detail"]) for r in rows if r["outcome"] not in ADMITTED
                               and r["outcome"] not in (REFUSED, REFUSED_UNNAMED))),
    }
    arm = ARMS.get(rows[0]["arm"]) if rows else None
    if arm is not None and arm.changes_a_field:
        controls = [r for r in counterexamples if "control" in r]
        attributable = [r for r in controls if r["control"] == EQUAL]
        out["teeth"] = len(attributable)
        out["teeth_basis"] = "off-arm counterexamples whose paired control (the field on) is equal"
        out["control_outcomes"] = dict(Counter(r["control"] for r in controls))
    elif arm is not None and arm.mechanism:
        out["teeth"] = sum(1 for r in counterexamples if arm.mechanism in r.get("classes", ()))
        out["teeth_basis"] = (f"counterexamples attributed to {arm.mechanism!r}: the base toolchain itself "
                              f"violates the hypothesis (no assertion site on this tree)")
    if "teeth" in out:
        out["teeth_rate_interval_95"] = clopper_pearson(out["teeth"], len(admitted))
    return out


def _kind(detail: str) -> str:
    head = detail.split(":", 1)
    return head[0] if head else detail


def network_relation(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per semantics, Proposition 2's relation on the identity programs: instances whose identity program
    departs from N, the neuron-windows compared and differing, and whether Proposition 2 claims N there (a
    conditional row is measured, never judged)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        claim = r.get("network_claim")
        if not claim or not r.get("network_windows"):
            continue
        cell = out.setdefault(r["semantics"], {"claim": claim, "instances": 0, "differ": 0, "windows": 0,
                                               "differing_windows": 0})
        cell["instances"] += 1
        cell["differ"] += int(r.get("network_mismatches", 0) > 0)
        cell["windows"] += int(r["network_windows"])
        cell["differing_windows"] += int(r.get("network_mismatches", 0))
    for cell in out.values():
        cell["upper_bound_95"] = one_sided_upper(cell["differ"], cell["instances"])
    return {"per_semantics": dict(sorted(out.items())),
            "claimed_and_differing": sorted(s for s, c in out.items() if c["claim"] == CLAIMED and c["differ"])}


def golden_relation(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """The law-carrying runs against G2's int64 golden at atol 0 (identity program and deployment of every admitted
    value instance): instances compared, output windows, mismatching windows and instances with any."""
    compared = [r for r in rows if r["outcome"] in ADMITTED and r.get("golden_windows")]
    return {"instances": len(compared), "windows": sum(int(r["golden_windows"]) for r in compared),
            "differing_windows": sum(int(r.get("golden_mismatches", 0)) for r in compared),
            "differ": sum(1 for r in compared if r.get("golden_mismatches")),
            "upper_bound_95": one_sided_upper(sum(1 for r in compared if r.get("golden_mismatches")), len(compared))}


def weights_and_dependence(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per semantics: per neural-layer index, all-zero layers and the largest weight magnitude drawn; and the
    input dependence of the identity programs (the fraction of neurons whose output varies over the batch)."""
    out: Dict[str, Any] = {}
    for sem in sorted({r["semantics"] for r in rows}):
        mine = [r for r in rows if r["semantics"] == sem]
        layers: Dict[int, Counter] = {}
        for r in mine:
            for index, peak in enumerate(r.get("weight_ranges", ())):
                layers.setdefault(index, Counter())[int(peak)] += 1
        dependent = [float(r.get("input_dependent", 0.0)) for r in mine]
        out[sem] = {
            "layers": {str(i): {"layers": sum(c.values()), "all_zero": c.get(0, 0),
                                "max_abs": dict(sorted(c.items()))} for i, c in sorted(layers.items())},
            "input_dependent_median": statistics.median(dependent) if dependent else None,
            "input_independent_instances": sum(1 for d in dependent if d == 0.0),
        }
    return out


def coverage(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per declared variable and motif: how many admitted instances exercised it."""
    admitted = [r for r in rows if r["outcome"] in ADMITTED]
    drawn = Counter(path for r in admitted for path in r.get("drawn", ()))
    motifs = Counter(m for r in admitted for m in r.get("motifs", ()))
    parts = partition()
    return {
        "admitted": len(admitted),
        "drawn_paths": {p: drawn.get(p, 0) for p in parts["program_changing"]},
        "derived_geometry": {p: "derived from cores.max_axons / cores.max_neurons, drawn through them"
                             for p in parts["derived_geometry"]},
        "drawn_by_fields": {p: f"drawn through its fields; present in {_present(admitted, p)} admitted declarations "
                               f"(where the registry's relevance holds)" for p in parts["drawn_by_fields"]},
        "pricing_only": {p: PRESERVED_BY_CONSTRUCTION for p in parts["pricing_only"]},
        "motifs": {m: motifs.get(m, 0) for m in MOTIFS},
        "semantics": dict(Counter(r["semantics"] for r in admitted)),
        "timesteps": dict(Counter(int(r.get("timesteps", 0)) for r in admitted)),
        "declared": _declared(admitted),
        "structure": {k: sum(1 for r in admitted if r.get(k)) for k in STRUCTURE},
        "weights_and_input_dependence": weights_and_dependence(admitted),
        "fixed_function_bearing_axes": dict(FIXED_AXES),
    }


def _present(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for r in rows if key in (r.get("declaration") or {}))


def _declared(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Distributions of the drawn declaration fields over the admitted instances."""
    decls = [r["declaration"] for r in rows if r.get("declaration")]
    cores = [c for d in decls for c in d.get("cores", ())]

    def bands(values: Iterable[int], width: int) -> Dict[str, int]:
        return dict(sorted(Counter(f"{(v - 1) // width * width + 1}-{(v - 1) // width * width + width}"
                                   for v in values).items()))

    return {
        "core_types": dict(Counter(len(d.get("cores", ())) for d in decls)),
        "max_axons": bands((c["max_axons"] for c in cores), 16),
        "max_neurons": bands((c["max_neurons"] for c in cores), 16),
        "count": bands((c["count"] for c in cores), 8),
        "max_schedule_passes": dict(Counter(d.get("max_schedule_passes") for d in decls)),
        "flags_true": {k: sum(1 for d in decls if d.get(k)) for k in
                       ("allow_coalescing", "allow_neuron_splitting", "allow_scheduling")},
        "core_value_granularity": dict(Counter(json_key(d.get("core_value_granularity")) for d in decls)),
        "residency_plan": dict(Counter(json_key(d.get("residency_plan")) for d in decls)),
    }


def json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


class ScreenTableError(ToolchainRefusal, ValueError):
    """A hypothesis table naming a code arm the arm registry does not hold."""


def denominators(table: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The denominator over every hypothesis of the box (the fixture's rows, in order): per row its code arm
    and how it bites (switched off with a paired control, or violated by the base and attributed by
    mechanism), its model arm, and where it is pending when neither level exercises it."""
    per: Dict[str, Dict[str, Any]] = {}
    for h in table["hypotheses"]:
        code, model = h["arms"]["code"], h["arms"]["model"]
        if code is not None and code not in ARMS:
            raise ScreenTableError(f"hypothesis {h['id']}: code arm {code!r} is not an arm of the registry")
        basis = None if code is None else "switched_off" if ARMS[code].changes_a_field else "violated_by_the_base"
        cl1 = sorted({s["site"].split(" ", 1)[0] for s in h["sites"] if s["tree"] == "CL1"})
        pending = None if code or model else f"CL1 ({', '.join(cl1)})" if cl1 else h.get("code_level") or h["gap"]
        per[h["id"]] = {"kind": h["kind"], "code_arm": code, "code_basis": basis, "model_arm": model,
                        "code_teeth": arms.get(code, {}).get("teeth") if code else None, "pending": pending}
    ids = list(per)
    switched = [i for i in ids if per[i]["code_basis"] == "switched_off"]
    violated = [i for i in ids if per[i]["code_basis"] == "violated_by_the_base"]
    exercisable = switched + violated
    with_teeth = [i for i in exercisable if (per[i]["code_teeth"] or 0) > 0]
    model_only = [i for i in ids if not per[i]["code_arm"] and per[i]["model_arm"]]
    neither = {i: per[i]["pending"] for i in ids if not per[i]["code_arm"] and not per[i]["model_arm"]}
    return {"hypotheses": ids, "per_hypothesis": per, "switched_off_with_paired_controls": switched,
            "violated_by_the_base": violated, "exercisable_at_code_level": exercisable,
            "not_exercisable_at_code_level": [i for i in ids if i not in exercisable],
            "with_teeth": with_teeth, "without_teeth": [i for i in exercisable if i not in with_teeth],
            "model_level_only": model_only, "not_exercised": neither,
            "restated": f"{len(switched)} switched off with paired controls + {len(violated)} violated by the "
                        f"base, of {len(ids)} hypotheses; {len(model_only)} at model level only; {len(neither)} "
                        f"at neither level ({', '.join(neither)})"}


def report(rows: Sequence[Mapping[str, Any]], *, table: Mapping[str, Any],
           model_level: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The screen report: per arm, the on arms pooled, the denominators over the hypothesis ``table`` (the
    theorem box's fixture) and the coverage of the on arms."""
    by_arm: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    arms = {arm: arm_summary(by_arm[arm]) for arm in sorted(by_arm)}
    on_rows = [r for a in ON_ARMS for r in by_arm.get(a, ())]
    shipped = [r for a in SHIPPED_ON_ARMS for r in by_arm.get(a, ())]
    harness = [r for a in ON_ARMS if ARMS[a].harness_setting for r in by_arm.get(a, ())]
    beyond = by_arm.get(ON_LONG_WINDOW, [])
    window = ARMS[ON_LONG_WINDOW].timesteps or (0, 0)
    return {
        "arms": arms,
        "hypothesis_on": arm_summary(shipped) if shipped else None,
        "hypothesis_on_scope": f"toolchain as shipped: {', '.join(SHIPPED_ON_ARMS)} (disjoint generator streams)",
        "hypothesis_on_harness_setting": arm_summary(harness) if harness else None,
        "hypothesis_on_harness_setting_scope": "value-keyed runtime residency classes (harness setting: "
                                               "legacy_convert._residency_class_of replaced)",
        "hypothesis_on_all_strata": arm_summary(on_rows) if on_rows else None,
        # Over every hypothesis-on seed whose identity program ran (refused deployments included): the
        # relation is the identity program's, and the three strata draw disjoint networks.
        "network_relation": network_relation(on_rows),
        # The value programs' integer readout law against G2's int64 golden: the on strata, and every arm's rows.
        "golden_relation": golden_relation(on_rows),
        "golden_relation_all_arms": golden_relation(rows),
        # Beyond the generator's declared window range: its own line, never pooled into the declared range's.
        "hypothesis_on_beyond_declared_window": arm_summary(beyond) if beyond else None,
        "hypothesis_on_beyond_declared_window_scope": (
            f"{ON_LONG_WINDOW}: scheduled synchronized LIF and TTFS deployments at T in {window[0]}-{window[1]}, "
            f"beyond the declared T in {DEFAULT_SPACE.timesteps[0]}-{DEFAULT_SPACE.timesteps[1]} (toolchain as "
            f"shipped, its own generator stream)"),
        "network_relation_beyond_declared_window": network_relation(beyond),
        "denominators": denominators(table, arms),
        "coverage": coverage(on_rows),
        "model_level": model_level,
    }
