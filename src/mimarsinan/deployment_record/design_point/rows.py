"""One judged chip row of an instance: placement, certificate, activity census, pass timeline, writes, lifetime, K*
and the admission verdict at the declared batch, from a built program and its priced platform.

The verdict: ``refused`` when the certificate finds a mismatch (a wrong output is never priced) or the write law
refuses the lifetime by name; otherwise the write law's lifetime verdict at the declared batch (``admitted``,
``inadmissible``, ``straddles``). A program that writes no cell per inference is admitted with that reason.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from mimarsinan.deployment_record.cost.absolute.writes import ENERGY_WRITE, LATENCY_WRITE
from mimarsinan.deployment_record.design_point.endurance import k_star, lifetime_at, priced_at, technology_wear
from mimarsinan.deployment_record.design_point.executors import ProgramCertificate, certify_program
from mimarsinan.deployment_record.design_point.placement import operator_of_nodes, operator_placement
from mimarsinan.deployment_record.design_point.timeline import pass_timeline, program_writes, writes_summary
from mimarsinan.deployment_record.design_point.wear import single_type_wear
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.layout_types import ROWS_UPPER_BOUND

CERTIFICATE_MISMATCH = "certificate_mismatch"


def program_summary(ir_graph: Any, hybrid: Any) -> Dict[str, Any]:
    neural = [s for s in hybrid.stages if s.kind == "neural"]
    return {"neural_passes": len(neural), "host_stages": sum(1 for s in hybrid.stages if s.kind == "compute"),
            "segments": len({s.schedule_segment_index for s in neural}),
            "hard_cores_used": max((len(s.hard_core_mapping.cores) for s in neural), default=0),
            "ir_cores": len(ir_graph.get_neural_cores()),
            "host_ops": sum(1 for n in ir_graph.nodes if not isinstance(n, NeuralCore))}


def admission_of(certificate: ProgramCertificate, lifetime: Mapping[str, Any]) -> Dict[str, Any]:
    """The row's verdict at the declared batch, its reasons and its refusals by name."""
    k = int(lifetime["at_K"])
    if not certificate.exact:
        wrong = [e.name for e in certificate.edges if e.mismatches] or ["per-operator census"]
        return {"verdict": "refused", "at_K": k, "reasons": [],
                "refusals": [{"name": CERTIFICATE_MISMATCH, "stage": "certify",
                              "reason": f"atol-0 mismatches on {', '.join(wrong)}"}]}
    reasons = [f"atol-0 integer certificate exact on {len(certificate.edges)} edges, {certificate.inputs} inputs"]
    if not certificate.nonvacuous:
        reasons.append("some observed operator has no nonzero output on an executor")
    if lifetime["verdict"] is None:
        return {"verdict": "refused", "at_K": k, "reasons": reasons,
                "refusals": [dict(r, stage="price") for r in lifetime["refusals"]]}
    return {"verdict": lifetime["verdict"], "at_K": k, "reasons": reasons + [lifetime["reason"]], "refusals": []}


def judged_row(*, nf: Any, ir_graph: Any, hybrid: Any, x_int: Any, platform: Mapping[str, Any],
               physics: PlatformPhysics, k_grid: Sequence[int], weight_bits: Optional[int],
               names: Optional[Mapping[int, str]] = None,
               x_fluid: Optional[float] = None, x_packed: Optional[float] = None,
               write_technology: Optional[str] = None,
               roles: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The judged fields of one row (``program``, ``placement``, ``timeline``, ``certificate``, ``activity``,
    ``writes``, ``lifetime``, ``k_star``, ``admission``) on ``platform`` at its declared ``inference_batch_K``, the
    payload bytes at the caller's ``weight_bits``. [E-6, targeted pass] On a chip whose core types declare several
    write technologies the write law prices each technology's share and each core type states its own wear
    (``technology_wear``; ``roles``: write physics name to the row's role of that core type)."""
    operator_of = operator_of_nodes(ir_graph, names)
    certificate = certify_program(nf, ir_graph, hybrid, x_int, names=names)
    writes = program_writes(hybrid, platform, weight_bits=weight_bits)
    summary = writes_summary(writes)
    declared = summary["declared_K"]
    pricing = priced_at(writes, physics, platform, declared)
    for name, key in ((ENERGY_WRITE, "energy_write_mj"), (LATENCY_WRITE, "latency_write_exposed_s")):
        band = next((t.band for t in pricing.terms if t.name == name and t.band is not None), None)
        summary[key] = None if band is None else [band.low, band.nominal, band.high]
    summary["refusals"] = [{"name": r.name, "reason": r.reason} for r in pricing.refusals]
    lifetime = lifetime_at(writes, physics, platform, declared)
    wear = technology_wear(writes, physics, declared, roles or {})
    if wear is None:
        wear = single_type_wear(summary, lifetime, "any", write_technology or platform.get("platform_physics_profile"))
    summary["per_core_type"], lifetime["per_core_type"], lifetime["binding"] = (
        wear["writes"], wear["lifetime"], wear["binding"])
    notes: List[str] = ["executors run the executed multi-pass program; references are the converted model (nf) "
                        "and the identity mapping (scm)"]
    return {
        "program": program_summary(ir_graph, hybrid),
        "placement": operator_placement(ir_graph, hybrid, names=names),
        "timeline": pass_timeline(hybrid, writes, operator_of),
        "certificate": certificate.to_dict(atol=0.0, notes=notes),
        "activity": [a.to_dict() for a in certificate.activity],
        "writes": summary, "lifetime": lifetime,
        "k_star": k_star(writes, physics, platform, k_grid, x_fluid=x_fluid, x_packed=x_packed),
        "admission": admission_of(certificate, lifetime),
    }


def _same(a: Any, b: Any, rtol: float) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and (isinstance(a, float) or isinstance(b, float)):
        return abs(float(a) - float(b)) <= rtol * max(abs(float(b)), 1e-300)
    return a == b


def plane_agreement(executed: Mapping[str, Any], candidate: Mapping[str, Any], rtol: float = 1e-9) -> Dict[str, Any]:
    """Field-by-field agreement of the executed program's census with the exploration plane's priced program (a float
    within ``rtol``: one pricer over equal census integers); a disagreement the instance keeps is named in ``named``,
    else ``validate_instance`` refuses the row. [U-AX] A candidate whose rows are an upper bound (its
    ``physical_rows``: a per-axon walk that read no weights) reads other rows than the executed program, so the gate
    compares only planes that read the same rows: each field such a candidate disagrees on is named with that cause."""
    fields = [{"field": name, "executed": executed.get(name), "candidate": candidate.get(name),
               "agree": _same(executed.get(name), candidate.get(name), rtol)} for name in executed]
    out: Dict[str, Any] = {"agree": all(f["agree"] for f in fields), "fields": fields}
    rows = candidate.get("physical_rows") or {}
    named = [{"field": f["field"], "reason": f"the candidate plane's rows are an upper bound ({rows.get('reason')}), "
              "the executed program's the deployed rows: the planes read different rows"}
             for f in fields if not f["agree"]] if rows.get("basis") == ROWS_UPPER_BOUND else []
    return {**out, "named": named} if named else out
