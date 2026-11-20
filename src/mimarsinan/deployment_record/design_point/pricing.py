"""A row's priced variants read from the exploration plane (the scaling ladder's priced point), and the comparison of
that plane's program with the executed program's census.

The exploration prices the program the search prices (the candidate plane). A variant's headlines are the write law's
at the variant's batch K (``write_law``: the energy and the latency with the writes of a period over K, refused by
name where the write law refuses them); the ladder's steady headlines without writes ride beside. The energy split
is the headline's own parts at K (``ENERGY_PARTS``): dynamic (MACs, periphery and carry; ``mac`` and ``carry`` are
its compute-only and boundary shares), static power over the priced latency, static writes, operand writes (the write
law's energy, apart only where the census separates them) and host; the host share is the nominal host part over the
nominal headline at K. A part the pricer leaves unpriced because its
constant is undeclared contributes nothing to the priced headline and reads 0; ``split_closes`` checks the sum
against the headline per corner. A ladder row that refused or ran above its budget becomes the pricing refusal by
name. [E-6, targeted pass] A row the exploration plane cannot price (operand cores) prices on the record plane:
``executed_record`` seals its executed program's record, ``record_pricing`` prices each variant at each batch.
[U-SH] Both planes price the ``mac`` and ``carry`` shares by one rule (``cost.absolute.host_split``); a null share
names its refusal in ``dynamic_shares_refusals``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.report import (
    absolute_pricing_for_record, declared_physics_of, record_compute_split, record_geometry, record_readout_bits,
    record_write_inputs,
)
from mimarsinan.deployment_record.cost.absolute.energy import ENERGY_DYNAMIC, ENERGY_HEADLINE
from mimarsinan.deployment_record.cost.absolute.host_split import (
    absent_shares, price_shares, share_column, shares, sub_corners,
)
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED
from mimarsinan.deployment_record.cost.absolute.writes import ENERGY_WRITE, price_write_inputs
from mimarsinan.deployment_record.build.from_mapping import (
    placement_record_from_mapping, schedule_record_from_mapping, utilization_record_from_mapping,
)
from mimarsinan.deployment_record.design_point.schema import ENERGY_PARTS
from mimarsinan.deployment_record.introspection.scaling_ladder_reading import write_law_columns
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.value_domain import names_value_domain
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FORMAT_VERSION, AccuracyReadRecord, AccuracyRecord, ComputePartitionRecord,
    DeploymentRecord, LatencyDecomposition, RecordIdentity, TimingRecord,
)
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for

PLANE = ("exploration (candidate plane: the scaling ladder's priced program at the row's budget, its census checked "
         "field by field against the executed program's census in ``planes``)")
#: Relative tolerance of a float fact's agreement between the planes (one pricer over equal census integers).
PLANE_RTOL = 1e-9
#: Relative tolerance of the split's closure per corner (float sums of the same bands).
SPLIT_RTOL = 1e-9
_ZERO = [0.0, 0.0, 0.0]


def _band(value: Any) -> Optional[List[float]]:
    return None if value is None else [float(v) for v in value]


def _terms(row: Mapping[str, Any], name: str) -> Optional[List[float]]:
    return _band(((row.get("energy") or {}).get("terms") or {}).get(name))


def _closes(parts: Mapping[str, Optional[List[float]]], total: Optional[List[float]]) -> Optional[bool]:
    if total is None or any(v is None for v in parts.values()):
        return None
    sums = [sum(v[i] for v in parts.values() if v is not None) for i in range(3)]
    return all(abs(a - b) <= SPLIT_RTOL * max(abs(b), 1e-300) for a, b in zip(sums, total))


def _headline(row: Mapping[str, Any], name: str) -> "tuple[Optional[List[float]], Optional[str]]":
    """(the headline at the row's K, its refusal): the write law's where the row priced it, else the steady one."""
    write_law = row.get("write_law") or {}
    steady = (row.get("headlines") or {}).get(name)
    steady_refusal = (row.get("headline_refusals") or {}).get(name)
    if not write_law.get("terms"):
        return _band(steady), steady_refusal
    value = write_law["terms"].get(name)
    if value is not None:
        return _band(value), None
    return None, (write_law.get("refusals") or {}).get(name) or steady_refusal or "refused by the write law"


def priced_variant(row: Mapping[str, Any], *, operand_cells: int) -> Dict[str, Any]:
    """One ladder row (one variant at one batch) in the instance's form; [U-SH] a null dynamic share names its refusal
    (``dynamic_shares_refusals``, present only where a share is null)."""
    write_law = row.get("write_law") or {}
    law = write_law.get("terms") or {}
    energy, energy_refusal = _headline(row, "energy_per_inference_mj")
    latency, latency_refusal = _headline(row, "e2e_latency_s")
    writes = _band(law.get("energy_write_mj"))
    priced = energy is not None
    split: Dict[str, Optional[List[float]]] = {
        "dynamic": _band(law.get("energy_dynamic_mj")),
        "static": _band(law.get("energy_static_mj")) or (_ZERO if priced else None),
        "static_write": (writes or (_ZERO if priced else None)) if not operand_cells else None,
        "operand_write": _ZERO if not operand_cells else None,
        "host": _band(law.get("energy_host_mj")) or (_ZERO if priced else None),
    }
    assert set(split) == set(ENERGY_PARTS)
    semantics = row.get("latency_semantics") or {}
    headlines = row.get("headlines") or {}
    verdicts = write_law.get("verdicts") or []
    absent = absent_shares(row.get("energy") or {})  # [U-SH] each null share's name, by the one share rule
    refused = {share: absent[term] for share, term in (("mac", "compute_only"), ("carry", "boundary"))
               if term in absent}
    return {
        "variant": str(row.get("variant") or "shipped"), "K": int(row["inference_batch_K"]),
        "energy_mj": energy, "energy_split_mj": split, "latency_s": latency,
        "refusals": {"energy_per_inference_mj": energy_refusal, "e2e_latency_s": latency_refusal},
        "split_closes": _closes(split, energy),
        "dynamic_shares_mj": {"mac": _terms(row, "compute_only"), "carry": _terms(row, "boundary")},
        **({"dynamic_shares_refusals": refused} if refused else {}),
        "energy_without_writes_mj": _band(headlines.get("energy_per_inference_mj")),
        "latency_without_writes_s": _band(headlines.get("e2e_latency_s")),
        "host_share": None if energy is None or split["host"] is None or not energy[1] else split["host"][1] / energy[1],
        "time_per_inference_s": semantics.get("time_per_inference_s"),
        "request_latency_s": semantics.get("request_latency_s"),
        "throughput_inferences_per_s": semantics.get("throughput_inferences_per_s"),
        "lifetime_verdict": next((v.get("verdict") for v in verdicts if v.get("name") == "lifetime_admissible"), None),
    }


def ladder_pricing(rows: Sequence[Mapping[str, Any]], *, operand_cells: int) -> Dict[str, Any]:
    """The row's ``pricing`` block from its ladder rows; the first refused or above-budget row names the refusal."""
    priced = [r for r in rows if r.get("headlines") is not None]
    out: Dict[str, Any] = {"plane": PLANE, "variants": [priced_variant(r, operand_cells=operand_cells) for r in priced]}
    stopped = next((r for r in rows if r.get("refusal") or r.get("above_budget")), None)
    if stopped is not None:
        mark = stopped.get("refusal") or stopped.get("above_budget") or {}
        out["refusal"] = {"name": str(mark.get("name", "above_budget")),
                          "reason": str(mark.get("reason", mark)),
                          "stage": str(mark.get("stage", stopped.get("stage", "price")))}
    return out


def executed_facts(judged: Mapping[str, Any], cores: int) -> Dict[str, Any]:
    """The executed program's census facts the exploration plane also states."""
    writes = judged["writes"]
    return {"cores": cores, "neural_passes": judged["program"]["neural_passes"],
            "written_passes": writes["written_passes"],
            "most_written_core_writes_per_period": writes["most_written_core_writes_per_period"],
            "payload_bytes_per_period": writes["payload_bytes_per_period"],
            "energy_write_mj": writes.get("energy_write_mj")}


def candidate_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The same facts from a ladder row at K = 1 (its passes, reprogrammed passes and bytes, and the census writes of
    the most-written cell per inference, which at K = 1 with no operand cells is the most-written core's period writes);
    [U-AX] and the rows a per-axon row states it planned (``physical_rows``), which ``plane_agreement`` reads."""
    writes = (row.get("k_star") or {}).get("max_writes_per_cell_per_inference")
    law = (row.get("write_law") or {}).get("terms") or {}
    return {"cores": row.get("cores"), "neural_passes": row.get("census_passes", row.get("passes")),
            "written_passes": row.get("reprogram_passes"),
            "most_written_core_writes_per_period": None if writes is None else int(round(float(writes))),
            "payload_bytes_per_period": row.get("reprogrammed_bytes_per_period"),
            "energy_write_mj": None if law.get("energy_write_mj") is None else [float(v)
                                                                               for v in law["energy_write_mj"]],
            **({"physical_rows": dict(row["physical_rows"])} if row.get("physical_rows") else {})}  # [U-AX]


# [E-6, targeted pass] The record plane of an executed instance program ---------------------------------------------
# The deployment record the pipeline seals from the same mapping, minus the simulator fragments, priced by the record
# plane's own pricer.
#
# A row whose program the exploration plane cannot price (operand cores: the candidate plane plans no operand regions)
# prices on the record plane, as Figure 1's stacked cost bars need: ``executed_record`` seals the fragments the
# deployment-record step builds from the hybrid mapping (``build.from_mapping``: the schedule with its operand regions
# per pass, the placement, the crossbar utilization and, where the caller states it, the on-chip/host partition) with
# the static timing census (no simulator ran: ``per_segment`` empty, ``compute_sim_time_s`` None) under the row's
# resolved platform; ``absolute_pricing_for_record`` then prices it with its own physics, its write-law inputs
# (per write technology, N2b) and its MAC events per core type (``compute_physics``, E-6). Accuracy is not read here
# (the read is the instance's certificate), so the accuracy fragment states none (a carried read of 0 samples).

#: The latency note of a record no simulator timed (the static census only).
STATIC_TIMING_NOTE = ("design-point instance: no simulator ran, so per_segment is empty and compute_sim_time_s is "
                      "None; the pricer's latency terms read the static census and the declared physics")


def executed_record(hybrid: Any, platform: Mapping[str, Any], *, weight_bits: int, label: str,
                    partition: Optional[ComputePartitionRecord] = None, timesteps: Optional[int] = None,
                    pass_transfer: Optional[str] = None) -> DeploymentRecord:
    """The record of ``hybrid`` on the resolved ``platform`` (its ``platform_physics_resolved`` the physics it is
    priced with), the fragments sealed from the mapping as the deployment-record step seals them. The carry census
    of a program cut into passes states its timesteps and transfer discipline: on a value-domain platform (``mvm``
    cores) one evaluation and the collapsed boundary (``run_pass_transfer``'s rule: mvm carries no spikes), unless
    the caller states them; a spiking program needs the caller's (the census refuses by name otherwise)."""
    bits = int(weight_bits)
    if names_value_domain(platform):
        timesteps = 1 if timesteps is None else timesteps
        pass_transfer = transfer_for(streamed=False) if pass_transfer is None else pass_transfer
    schedule = schedule_record_from_mapping(hybrid, weight_bits=bits, params_reloaded=0, timesteps=timesteps,
                                            pass_transfer=pass_transfer)
    utilization = utilization_record_from_mapping(
        hybrid, crossbar_report=CrossbarUtilizationReport.from_hybrid_mapping(hybrid, weight_bits=bits),
        relay_cores_inserted=0, partition=partition,
        declared_core_types=core_types_from_declaration(list(platform.get("cores") or ())))
    declared = dict(platform)
    identity = RecordIdentity(
        format_version=DEPLOYMENT_RECORD_FORMAT_VERSION, run_dir="", cell_key=str(label), mode="design_point_instance",
        model_type="instance", model_name=str(label), workload=str(label),
        config_digest=hashlib.sha256(json.dumps(declared, sort_keys=True, default=str).encode()).hexdigest()[:16],
        platform=declared, deployment_options={}, created_at="")
    timing = TimingRecord(
        s_global=1, depth=len(schedule.segments()), per_segment=(),
        latency=LatencyDecomposition(programming_s=None, compute_steps=0, compute_sim_time_s=None,
                                     host_ops_s_raw_total=None, host_ops_s_per_pass=None, sync_s=None,
                                     note=STATIC_TIMING_NOTE))
    accuracy = AccuracyRecord(
        deployed=AccuracyReadRecord(metric=0.0, backend="value_census", samples=0, kind="carried",
                                    step="design_point_instance"),
        reads=(), certificates=())
    return DeploymentRecord(identity=identity, schedule=schedule, placement=placement_record_from_mapping(hybrid),
                            utilization=utilization, accuracy=accuracy, timing=timing, traffic=None, energy=None,
                            adaptation=None)


def record_at_batch(record: DeploymentRecord, k: int) -> DeploymentRecord:
    """The record with its platform's declared batch ``inference_batch_K`` set to ``k`` (the write law's K)."""
    return replace(record, identity=replace(record.identity, platform={**record.identity.platform,
                                                                       "inference_batch_K": int(k)}))



RECORD_PLANE = ("record (the executed program's record, sealed from its mapping as the deployment-record step seals "
                "it, without simulator fragments, priced by absolute_pricing_for_record: each core type's MAC events "
                "at its own e_mac, the write law per write technology; the static writes apart from the operand writes "
                "by the same law on the census without its operand regions)")


def _without_operand_writes(inputs: Any) -> Any:
    """The write-law inputs with every operand region's writes removed (the static writes alone, per technology)."""
    def strip(census: Any) -> Any:
        return replace(census, passes=tuple(replace(p, written_cells_producer=0) for p in census.passes))

    split = inputs.by_technology
    if split is not None:
        split = replace(split, parts=tuple(replace(part, census=strip(part.census), producer_slots=None)
                                           for part in split.parts))
    return replace(inputs, census=strip(inputs.census), producer_slots=None, by_technology=split)


def _write_parts(record: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """(static writes, operand writes) of the record's write energy at its batch, each None where refused."""
    inputs, physics = record_write_inputs(record), declared_physics_of(record)
    if inputs is None or inputs.census is None or physics is None:
        return None, None
    total = next((t.band for t in price_write_inputs(inputs, physics).terms if t.name == ENERGY_WRITE), None)
    static = next((t.band for t in price_write_inputs(_without_operand_writes(inputs), physics).terms
                   if t.name == ENERGY_WRITE), None)
    if total is None or static is None:
        return None, None
    return _band([static.low, static.nominal, static.high]), list(sub_corners(total, static, "operand writes"))


def record_variant(name: str, record: Any, k: int) -> Dict[str, Any]:
    """One variant at batch ``k`` on the record plane, in the instance's form (``priced_variant``'s fields): the
    headlines with the write law at K (``absolute_pricing_for_record``), the steady headlines without writes, and the
    energy split into its parts, the writes split into static and operand writes. [U-SH] The dynamic part's MAC and
    carry shares by the exploration plane's one share rule (``host_split.price_shares`` and ``shares``) on the
    record's quantities at K, with the variant's physics, compute split, readout bits and geometry."""
    at_k = record_at_batch(record, k)
    priced = absolute_pricing_for_record(at_k)
    physics = declared_physics_of(at_k)
    if priced is None or physics is None:
        return {"variant": name, "K": int(k), "energy_mj": None, "energy_split_mj": {p: None for p in ENERGY_PARTS},
                "latency_s": None, "refusals": {"energy_per_inference_mj": "the record declares no physics",
                                                "e2e_latency_s": "the record declares no physics"}}
    geometry = record_geometry(at_k)
    quantities, judged = from_record(at_k), UNJUDGED if geometry is None else geometry
    args = {"compute": record_compute_split(at_k), "readout_bits": record_readout_bits(at_k), "geometry": judged}
    steady = price_absolute(quantities, physics, **args)
    steady_terms = {t.name: t.band for t in steady.terms if t.band is not None}
    law = write_law_columns(priced, int(k))
    split = shares(*price_shares(quantities, physics, **args), ENERGY_DYNAMIC, ENERGY_HEADLINE)
    row = {"variant": name, "inference_batch_K": int(k), "write_law": law,
           "headlines": {n: None if steady_terms.get(n) is None else [steady_terms[n].low, steady_terms[n].nominal,
                                                                       steady_terms[n].high]
                         for n in ("energy_per_inference_mj", "e2e_latency_s")},
           "headline_refusals": {r.name: r.reason for r in steady.refusals}, "energy": share_column(split)}
    out = priced_variant(row, operand_cells=0)
    static_write, operand_write = _write_parts(at_k)
    energy = out["energy_mj"]
    out["energy_split_mj"] = {**out["energy_split_mj"], "static_write": static_write, "operand_write": operand_write}
    if energy is not None and (static_write is None or operand_write is None):
        out["energy_split_mj"]["static_write"] = out["energy_split_mj"]["operand_write"] = None
    out["split_closes"] = _closes(out["energy_split_mj"], energy)
    return out


def record_pricing(records: Mapping[str, Any], k_grid: Sequence[int]) -> Dict[str, Any]:
    """The row's ``pricing`` block on the record plane: every variant (label to its executed record, priced with
    its own resolved physics) at every batch of the grid."""
    return {"plane": RECORD_PLANE,
            "variants": [record_variant(name, record, int(k)) for name, record in records.items() for k in k_grid]}
