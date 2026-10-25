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
name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from mimarsinan.deployment_record.design_point.schema import ENERGY_PARTS

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
    """One ladder row (one variant at one batch) in the instance's form."""
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
    return {
        "variant": str(row.get("variant") or "shipped"), "K": int(row["inference_batch_K"]),
        "energy_mj": energy, "energy_split_mj": split, "latency_s": latency,
        "refusals": {"energy_per_inference_mj": energy_refusal, "e2e_latency_s": latency_refusal},
        "split_closes": _closes(split, energy),
        "dynamic_shares_mj": {"mac": _terms(row, "compute_only"), "carry": _terms(row, "boundary")},
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
    the most-written cell per inference, which at K = 1 with no operand cells is the most-written core's period writes)."""
    writes = (row.get("k_star") or {}).get("max_writes_per_cell_per_inference")
    law = (row.get("write_law") or {}).get("terms") or {}
    return {"cores": row.get("cores"), "neural_passes": row.get("census_passes", row.get("passes")),
            "written_passes": row.get("reprogram_passes"),
            "most_written_core_writes_per_period": None if writes is None else int(round(float(writes))),
            "payload_bytes_per_period": row.get("reprogrammed_bytes_per_period"),
            "energy_write_mj": None if law.get("energy_write_mj") is None else [float(v)
                                                                               for v in law["energy_write_mj"]]}
