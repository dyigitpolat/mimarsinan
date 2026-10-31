"""Reading columns of one scaling-ladder point beside the capacity law [CL5]: reuse per write, the two-ceiling
minimum batch and the latency semantics of a batch, as pure functions of priced numbers; and the ladder's one
period-write source (``period_writes``: G3's candidate-plane period census of the priced program).

- Stationary weights (``stationary_weight_cells``, [fix_b]): the logical weight cells of x_fluid and R, one count per
  network: per perceptron with on-chip softcores, its weight tensor once plus one bias row per output, independent of
  the target's geometry and bias capability; the layout's per-position copies are replication.
- Reuse per write (``reuse_columns``): R = on-chip MACs per inference / logical weight cells (MACs per weight), K x R
  the reuse each weight write buys at batch K, and Pi x R per axis, the write-to-MAC cost ratio of one weight
  (Pi / K = that ratio / (K x R)), so a point can be placed on a roofline-like map with a network term.
- Minimum batch (``min_batch``): a beyond-capacity program needs K at least the endurance ceiling and at least the
  write-bandwidth ceiling. Endurance: K >= m r T / E with m the writes per cell per inference at K = 1 (the census
  form, from the executed rotation) beside the closed forms (x - 1) r T / E at x_fluid and at x_packed (the rotation
  assumption). Bandwidth: one period of K inferences must write its swapped tiles within K / r, so K >= r t_w with
  t_w the write time of one period. E_cross = T m / t_w is the endurance at which the two ceilings meet. Corners pair
  the endurance band's high end with the write time's low end (favourable) and the reverse (adverse).
- Latency semantics (``latency_semantics``): the priced time per inference is the batch period over K; the batch
  period, the request latency (collecting K requests at rate r, then one period), the throughput, the carry buffer a
  batch holds and where it lives, and for a resident program (no pass rewrites cores) a layer-pipelined bound.

Corner lists are corner-aligned (low, nominal, high of the priced bands), not re-sorted bands.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.writes import ENERGY_WRITE, LATENCY_WRITE, WriteInputs, price_write_inputs
from mimarsinan.deployment_record.cost.host_terms import HOST_DECOMPOSITION
from mimarsinan.deployment_record.introspection.scaling_ladder_law import layer_pipelined, write_law_absence
from mimarsinan.deployment_record.quantities import write_technology as wt
from mimarsinan.deployment_record.quantities.write_law import (
    PlaneWrites, WritePolicy, plane_write_census, write_policy_of, write_quantities,
)

TIME_PER_INFERENCE = "time per inference (batch period / K)"
CARRY_HOME = ("read out of the arrays after its producing pass and re-injected before its consumer "
              "(carry_readout, carry_inject): held off the arrays, in the buffer the transfer reads; no on-chip carry "
              "buffer is declared")
CORNER_PAIRS = ("favourable", "nominal", "adverse")
NO_WRITES = "neither (no weight writes per period)"
_S_PER_H = 3600.0
Corners = Sequence[float]


def stationary_weight_cells(mapper_repr: Any, softcores: Sequence[Any]) -> int:
    """Per perceptron whose node lays out on-chip softcores (the specs' ``perceptron_index``), its weight tensor's
    elements plus one bias row per output (``weight.shape[0]``), summed: the cells a program writes once per weight
    bank on any geometry. Wiring cores without a perceptron hold no network weight."""
    onchip = {int(s.perceptron_index) for s in softcores if getattr(s, "perceptron_index", None) is not None}
    cells, seen = 0, set()
    for node in mapper_repr.execution_order():
        if getattr(node, "perceptron_index", None) not in onchip or not hasattr(node, "owned_perceptron_groups"):
            continue
        for perceptron in (p for group in node.owned_perceptron_groups() for p in group):
            if id(perceptron) not in seen:
                seen.add(id(perceptron))
                weight = perceptron.layer.weight
                cells += int(weight.numel()) + int(weight.shape[0])
    return cells


def reuse_columns(*, onchip_macs: Optional[float], logical_cells: int, k: Optional[int],
                  pi_by_axis: Mapping[str, Optional[Sequence[float]]]) -> Dict[str, Any]:
    """R (MACs per weight), K x R and Pi x R per axis."""
    if onchip_macs is None or not logical_cells:
        return {"macs_per_weight": None, "reuse_per_write": None, "write_to_mac_ratio": {},
                "absent": "onchip_macs or the logical weight cells are absent"}
    r = float(onchip_macs) / float(logical_cells)
    return {"macs_per_weight": r, "reuse_per_write": None if not k else k * r,
            "write_to_mac_ratio": {axis: None if pi is None else [p * r for p in pi]
                                   for axis, pi in pi_by_axis.items()}}


def _verdict(admits: List[bool]) -> str:
    return "admitted" if all(admits) else "refused" if not any(admits) else "straddles"


def min_batch(*, x_fluid: float, x_packed: float, max_writes: Optional[float], rate_hz: Optional[float],
              lifetime_hours: Optional[float], endurance: Optional[Corners],
              write_time_per_period: Optional[Corners], write_time_source: str, k: Optional[int],
              pulses_per_write: Optional[Corners] = None) -> Dict[str, Any]:
    """The endurance and write-bandwidth ceilings on K, their crossover endurance, and the verdict at K. [U-C C-1]
    With ``pulses_per_write`` the ceilings read endurance in writes (low with most pulses), crossovers in cycles at
    each write-time corner's own pulses (one iteration count sets both), so a corner's binding and crossover agree."""
    out: Dict[str, Any] = {"write_time_source": write_time_source, "inference_batch_K": k}
    p = None if pulses_per_write is None else [float(v) for v in pulses_per_write]
    if p is not None:
        out["pulses_per_write"] = p
    if rate_hz is None or lifetime_hours is None:
        missing = [n for n, v in (("inference_rate_hz", rate_hz), ("min_lifetime_hours", lifetime_hours)) if v is None]
        out.update(absent="undeclared: " + ", ".join(missing), verdict={"both": "undetermined"})
        return out
    life_s = lifetime_hours * _S_PER_H
    service = rate_hz * life_s
    e = None if endurance is None else [float(v) / f for v, f in zip(endurance, p[::-1] if p else (1.0,) * 3)]
    t = None if write_time_per_period is None else [float(v) for v in write_time_per_period]
    closed = {"fluid": x_fluid, "packed": x_packed}
    absent_closed = [f"x_{n} <= 1" for n, x in closed.items() if x <= 1.0]
    if absent_closed:
        out["closed_absent"] = "the closed form needs x > 1: " + ", ".join(absent_closed)
    for name, x in closed.items():
        out[f"k_endurance_closed_{name}"] = None if e is None or x <= 1.0 else [(x - 1.0) * service / v for v in e]
    if e is None or max_writes is None:
        out["k_endurance_census"] = None
        out["endurance_absent"] = ("endurance_cycles undeclared" if e is None
                                   else "the census states no writes per cell per inference")
    else:
        out["k_endurance_census"] = [max_writes * service / v for v in e]
    if t is None:
        out["k_bandwidth"] = None
        out["bandwidth_absent"] = "the write time of one period is not priced"
    else:
        out["k_bandwidth"] = [rate_hz * v for v in t]
        q = p if p else [1.0] * len(t)  # write-time corner i with pulses corner i: both read one iteration count
        out["e_cross_census"] = (None if max_writes is None else
                                 [life_s * max_writes * f / v if v > 0.0 else None for v, f in zip(t, q)])
        out["e_cross_closed_packed"] = (None if x_packed <= 1.0 else
                                        [life_s * (x_packed - 1.0) * f / v if v > 0.0 else None for v, f in zip(t, q)])
    end, bw = out["k_endurance_census"], out["k_bandwidth"]
    if end is not None and bw is not None:
        # favourable: highest endurance (end[2]) with the lowest write time (bw[0]); adverse the reverse
        pairs = {"favourable": (end[2], bw[0]), "nominal": (end[1], bw[1]), "adverse": (end[0], bw[2])}
        out["k_min_census"] = {c: max(a, b) for c, (a, b) in pairs.items()}
        out["binding"] = {c: NO_WRITES if a == 0.0 and b == 0.0 else "endurance" if a >= b else "write bandwidth"
                          for c, (a, b) in pairs.items()}
    verdict: Dict[str, str] = {}
    verdict["endurance"] = "undetermined" if end is None or not k else _verdict([k >= v for v in end])
    verdict["bandwidth"] = "undetermined" if bw is None or not k else _verdict([k >= v for v in bw])
    both = [verdict["endurance"], verdict["bandwidth"]]
    verdict["both"] = ("undetermined" if "undetermined" in both else "refused" if "refused" in both
                       else "straddles" if "straddles" in both else "admitted")
    out["verdict"] = verdict
    return out


def latency_semantics(*, time_per_inference: Optional[Corners], compute_only: Optional[Corners], k: Optional[int],
                      rate_hz: Optional[float], carry: Mapping[str, Any], passes: int,
                      latency_steps: Optional[int], multiplexed: bool, source: str,
                      compute_law: Optional[str] = None,
                      event_stage: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Batch period, request latency, throughput, carry buffer per K and the layer-pipelined bound (``compute_law``:
    the physics' compute-time law, ``cost.compute_time``; ``event_stage``: on several event-serial cores, the most
    loaded core's operations of one pass, ``PricedProgram.event_stage``)."""
    kk = int(k or 1)
    out: Dict[str, Any] = {"time_per_inference_label": TIME_PER_INFERENCE, "time_per_inference_source": source,
                           "time_per_inference_s": None if time_per_inference is None else list(time_per_inference)}
    if time_per_inference is None:
        out.update(batch_period_s=None, request_latency_s=None, throughput_inferences_per_s=None,
                   absent="the time per inference is not priced")
    else:
        period = [kk * v for v in time_per_inference]
        out["batch_period_s"] = period
        out["throughput_inferences_per_s"] = [1.0 / v if v > 0.0 else None for v in time_per_inference]
        if rate_hz:
            out["request_latency_s"] = [kk / rate_hz + p for p in period]
            out["request_latency_basis"] = "K / inference_rate_hz (collecting the batch) + the batch period"
        else:
            out["request_latency_s"] = None
            out["request_latency_absent"] = "inference_rate_hz undeclared"
    live = carry.get("peak_live_bytes", carry.get("carry_peak_live_bytes"))
    out["carry_buffer_bytes"] = None if live is None and carry else int(live or 0) * kk
    out["carry_buffer_home"] = CARRY_HOME
    out.update(layer_pipelined(compute_only, latency_steps, multiplexed=multiplexed, passes=passes,
                               compute_law=compute_law, event_stage=event_stage))
    return out


_MAX_WRITES = "max_writes_per_cell_per_inference"
_CENSUS_SOURCE = "the period census quantity (write_census, the executed rotation) at K = 1"
#: [CL7] The parts the energy headline at K is the corner-wise sum of (``cost/absolute/energy.py``): dynamic, static
#: power over the priced latency (exposed write time included), host and writes.
ENERGY_HEADLINE_PARTS = ("energy_dynamic_mj", "energy_static_mj", "energy_host_mj", "energy_write_mj")
WRITE_LAW_TERMS = ("energy_write_mj", "latency_write_exposed_s", "lifetime_hours", "lifetime_inferences",
                   "e2e_latency_s", "energy_per_inference_mj", "throughput_inferences_s",
                   *ENERGY_HEADLINE_PARTS[:3], *HOST_DECOMPOSITION)  # [U-J] the host's breakdown at K


def period_writes(passes: Any, platform: Mapping[str, Any], bits: Any) -> Tuple[int, Optional[int],
                                                                                Tuple[Optional[float], str]]:
    """(passes rewritten per period, period rewrite bytes, (max writes per cell per inference at K = 1, source)):
    G3's candidate-plane period census quantities of the priced program at the platform's weight bits ``bits``
    (read by the caller), the ladder's one write source."""
    if passes is None:
        return 0, None, (None, "the candidate plane states no pass structure, so it has no period census")
    census, reason = plane_write_census(passes, platform, weight_bits=bits, timesteps=None, label="candidate")
    if census is None:
        return 0, None, (None, f"the period census refuses: {reason}")
    values = write_quantities(census.census, WritePolicy(batch_k=1), written_extents=census.written_extents,
                              most_written=census.most_written_core)
    written = sum(1 for p in census.census.passes if not p.resident)
    wear = values.get(_MAX_WRITES)
    source = (f"{_CENSUS_SOURCE}{'' if wear is not None else ': no rewritten core is placed on a physical core'}"
              + (f" ({census.census.label})" if census.census.label else ""))
    return written, int(values["period_rewrite_bytes"].value), (None if wear is None else wear.value, source)


#: [consts] The write law's terms of a tile-write census that the ladder's axes read (energy, then time).
TILE_WRITE_TERMS = (ENERGY_WRITE, LATENCY_WRITE)


def tile_writes(passes: Any, platform: Mapping[str, Any], bits: Any, physics: Any, *,
                load: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """[consts] (terms, refusals by name) of the write law (``price_write_inputs``, the pricer of every per-inference
    write term) on the priced program's static tiles at K = 1, serial: ``load`` its deployment load, every core of
    every pass written once (``period_census(every_core=True)``; Pi's numerator, law.md §2.1), else its own period
    census (M's period writes). Producer writes are per inference, not tile writes, and are left out. One source for
    what writing a tile costs: a write-verify wall, a cell commit energy and the per-core e_core_program enter
    wherever the law prices them, and where the law refuses, so does the term, by the law's reason."""
    if passes is None:
        reason = "the candidate plane states no pass structure, so it has no tile-write census"
        return {}, {name: reason for name in TILE_WRITE_TERMS}
    rows = tuple(replace(row, producer_cells=0, producer_cores=frozenset(), producer_core_cells=None)
                 for row in passes.rows)
    program = replace(passes, rows=rows, operands_unstated=None)
    policy = write_policy_of({**platform, "inference_batch_K": 1})
    assert policy is not None
    policy = replace(policy, overlap_mode="serial", producer_write_source=None, payload_bits=None)
    census, reason = plane_write_census(program, platform, weight_bits=bits, timesteps=None, every_core=load,
                                        label="candidate tiles, load" if load else "candidate tiles, period")
    split = None if census is None or policy.core_technologies is None else wt.split_by_technology(
        program, census, policy, weight_bits=int(bits))
    pricing = price_write_inputs(WriteInputs.of(PlaneWrites(policy, census, reason, split)), physics)
    return ({t.name: t.band for t in pricing.terms if t.band is not None and t.name in TILE_WRITE_TERMS},
            {r.name: r.reason for r in pricing.refusals if r.name in TILE_WRITE_TERMS})


def _triple(band: Any) -> Optional[List[float]]:
    return None if band is None else [band.low, band.nominal, band.high]


def write_law_columns(pricing: Any, k: Any) -> Dict[str, Any]:
    """The write law's terms, refusals and admissibility verdicts priced at batch ``k`` (G3's ``price_absolute``
    with the period census), with the headlines at that batch."""
    terms = {t.name: t.band for t in pricing.terms if t.band is not None}
    refused = {r.name: r.reason for r in pricing.refusals}
    return {"inference_batch_K": k, "terms": {n: _triple(terms.get(n)) for n in WRITE_LAW_TERMS},
            "refusals": {n: refused[n] for n in WRITE_LAW_TERMS if n in refused},
            "verdicts": [{"name": v.name, "verdict": v.verdict, "reason": getattr(v, "reason", "")}
                         for v in getattr(pricing, "verdicts", ())]}


def _write_time_per_period(priced: Any, axes: Mapping[str, Any], k: Optional[int]) -> Tuple[Optional[List[float]], str]:
    exposed = (priced.write_law.get("terms") or {}).get("latency_write_exposed_s") if priced.write_law else None
    if exposed is not None and k:
        return [v * k for v in exposed], ("the write law's exposed write time at K, times K (serial: the write "
                                          "time of one period)")
    period = (axes.get("time") or {}).get("terms", {}).get("period")
    if period is not None:
        return list(period), ("the write law's serial write time of the period census at K = 1 (tile_writes); its "
                              "exposed write time at K is refused or undeclared")
    return None, "no write time is priced"


def _time_per_inference(priced: Any, axes: Mapping[str, Any], k: Optional[int]) -> Tuple[Optional[List[float]], str]:
    headline = (priced.write_law.get("terms") or {}).get("e2e_latency_s") if priced.write_law else None
    if headline is not None:
        return list(headline), "e2e_latency_s priced with the write law at K"
    time = (axes.get("time") or {}).get("terms", {})
    steady, period = priced.terms.get("program", {}).get("e2e_latency_s"), time.get("period")
    if steady is None:
        return None, "e2e_latency_s is refused"
    tpi = [steady.low, steady.nominal, steady.high]
    if period is not None and k:
        tpi = [a + b / k for a, b in zip(tpi, period)]
    return tpi, ("steady-state e2e_latency_s plus the period's serial write time / K (tile_writes: the writes "
                 "amortized over the batch)")


def endurance_fields(physics: Any) -> Dict[str, Any]:
    """A priced program's endurance fields (``PricedProgram``): the declared cycles (nominal and band) and the pulses
    per write where the technology declares them; none without physics or an endurance."""
    cycles, pulses = (None if physics is None or not physics.has(k) else tuple(getattr(physics.band(k), c) for c in (
        "low", "nominal", "high")) for k in ("endurance_cycles", "pulses_per_write"))
    return {"endurance": None if cycles is None else cycles[1], "endurance_band": cycles,
            "pulses_band": pulses if cycles is not None else None}


def row_readings(priced: Any, *, x_fluid: float, x_packed: float, k: Optional[int], logical_cells: int,
                 onchip_macs: Optional[float], rate_hz: Optional[float], lifetime_hours: Optional[float],
                 axes: Mapping[str, Any], multiplexed: bool) -> Dict[str, Any]:
    """The CL5 reading columns of one ladder row: write law at K, reuse per write, minimum batch, latency semantics."""
    t_w, t_w_source = _write_time_per_period(priced, axes, k)
    tpi, tpi_source = _time_per_inference(priced, axes, k)
    compute = (axes.get("time") or {}).get("terms", {}).get("compute_only")
    return {
        "write_law": dict(priced.write_law) or {"absent": write_law_absence(priced.terms)},
        "reuse": reuse_columns(onchip_macs=onchip_macs, logical_cells=logical_cells, k=k,
                               pi_by_axis={a: (axes.get(a) or {}).get("pi") for a in axes}),
        "min_batch": min_batch(x_fluid=x_fluid, x_packed=x_packed, max_writes=priced.max_writes[0], rate_hz=rate_hz,
                               lifetime_hours=lifetime_hours, endurance=priced.endurance_band,
                               write_time_per_period=t_w, write_time_source=t_w_source, k=k,
                               pulses_per_write=priced.pulses_band),
        "latency_semantics": latency_semantics(time_per_inference=tpi, compute_only=compute, k=k, rate_hz=rate_hz,
                                               carry=dict(priced.carry or {}), passes=priced.passes,
                                               latency_steps=priced.latency_steps, multiplexed=multiplexed,
                                               source=tpi_source, compute_law=priced.compute_law,
                                               event_stage=priced.event_stage),
    }
