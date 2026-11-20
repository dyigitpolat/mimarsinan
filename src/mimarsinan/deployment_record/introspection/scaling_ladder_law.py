"""The capacity-law columns of one scaling-ladder point, as pure functions of priced bands.

Definitions (`06_parts/law.md` §1-§3; `lens_arch.md` §2.2-§2.3):
- x_fluid = logical weight cells / capacity cells; x_packed = packed tiles / cores.
- Two readings per point. The capacity reading compares the packed tiles with the
  cores (x_packed <= 1: the program fits by capacity). The schedule reading is the
  priced program's period census (G3's ``write_census``, the executed rotation): a
  program whose passes rewrite cores every period is multiplexed, whatever x_packed
  is, and one that rewrites none writes each cell once per deployment (resident).
  The regime, the period writes in M and the per-inference write census follow the
  schedule; M is printed for every program of two or more passes, so a resident
  multi-pass program (residency passes after its load) prices M with no period
  writes.
- Pi = cost of writing every tile of the program once / compute-only cost of one
  inference, per axis (energy, time); Pi/K at the declared batch K. [consts] The numerator
  is the one load law's (``cost/absolute/walls/loads.py``: the program's deployment load at
  K = 1, serial), the pricer of every per-inference write, so its commit energy, its
  per-core e_core_program and its write-verify wall are Pi's; where the law refuses, the
  axis is absent with its reason. [U-ENERGY] In energy each cost carries the chip's static
  power over its own wall (the load over its serial wall, one inference over its compute-only time).
  M's period writes are the same law on the program's period census.
- Regime per band corner of Pi/K under caller-declared reading thresholds (the
  thresholds are reading constants of the caller, never framework constants);
  "straddles" when the corners carry different labels; a resident program reads
  RESIDENT.
- M = priced cost over compute-only cost; the closed forms 1 + phi(x, Q) Pi/K + kappa
  (residency floor, phi = 0 when the program fits, x <= 1) and 1 + Pi/K + kappa
  (stateless reload), kappa = boundary cost (carry and sync) over compute-only cost.
  Each is printed per corner: a corner prices every term at that corner's
  constants, so the corner triples are aligned, not bands, and need not be ordered.
- K*_closed = (x_fluid - 1) r T / E (rotation assumption), K*_census =
  r T max_writes / E, and their ratio split exactly into (x_packed - 1)/(x_fluid - 1)
  (packing and replication) and max_writes/(x_packed - 1) (rotation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, div_band, scale_band
from mimarsinan.deployment_record.cost.absolute.host_split import (  # [U-SH] the share rule, one for every plane
    NO_PHYSICS as NO_PHYSICS, share_column, shares as shares, sub_corners as sub_corners,
)
from mimarsinan.deployment_record.cost.compute_time import EVENT_LAW
from mimarsinan.deployment_record.schema.provenance import Band

CORNERS = ("low", "nominal", "high")
RESIDENT = "resident (no per-inference weight writes)"
STRADDLES = "straddles"
FITS_BY_CAPACITY = "fits by capacity (x_packed <= 1)"
CORNER_NOTE = "corner-aligned: each corner prices every term at that corner's constants (not a band)"
_SECONDS_PER_HOUR = 3600.0
#: Low, nominal and high values, each priced at that corner's constants.
Corners = Tuple[float, float, float]


@dataclass(frozen=True)
class RegimeReading:
    """Reading thresholds on Pi/K, declared by the caller with their source."""

    capacity_free_max: float
    resident_dominant_min: float
    source: str

    def label(self, pi_over_k: float) -> str:
        if pi_over_k <= self.capacity_free_max:
            return "capacity-free"
        if pi_over_k >= self.resident_dominant_min:
            return "resident-dominant"
        return "informative"


def capacity_reading(x_packed: float) -> str:
    return FITS_BY_CAPACITY if x_packed <= 1.0 else f"exceeds capacity (x_packed = {x_packed:.6g})"


def schedule_reading(written_passes: int, segments: int) -> str:
    """The schedule reading of a program whose period census rewrites ``written_passes`` passes."""
    if written_passes:
        return f"U1: multiplexed by {written_passes} reprogrammed passes over {segments} neural segments"
    return (f"resident: no pass rewrites its cores per period over {segments} neural segments, "
            "weights written once per deployment")


def triple(band: Optional[Band]) -> Optional[list]:
    return None if band is None else [band.low, band.nominal, band.high]


def corners(band: Band) -> Corners:
    return (band.low, band.nominal, band.high)


def ratio(numerator: Optional[Band], denominator: Optional[Band], basis: str) -> Optional[Band]:
    if numerator is None or denominator is None or denominator.low <= 0.0:
        return None
    return div_band(numerator, denominator, basis)


def share(part: Optional[Band], total: Optional[Band]) -> Optional[float]:
    """Nominal part over nominal total (the host share of a priced headline), or None."""
    return None if part is None or total is None or total.nominal <= 0.0 else part.nominal / total.nominal


def per_batch(pi: Optional[Band], k: Optional[int]) -> Optional[Band]:
    return None if pi is None or not k else scale_band(pi, 1.0 / k, f"({pi.basis}) / K")


def regime(pi_over_k: Optional[Band], reading: RegimeReading, multiplexed: bool) -> Optional[Dict[str, str]]:
    if pi_over_k is None:
        return None
    labels = {corner: reading.label(v) for corner, v in zip(CORNERS, triple(pi_over_k) or ())}
    distinct = set(labels.values())
    labels["reading"] = (RESIDENT if not multiplexed else next(iter(distinct)) if len(distinct) == 1
                         else STRADDLES)
    return labels


def phi(x: float, passes: int) -> Optional[float]:
    """Written fraction per batch at the residency floor, Q(x-1)/((Q-1)x) (law.md L1); 0 when x <= 1."""
    if x <= 1.0:
        return 0.0
    if passes < 2:
        return None
    return passes * (x - 1.0) / ((passes - 1) * x)


def _rel(values: List[float], references: List[float]) -> List[float]:
    return [(v - r) / r for v, r in zip(values, references)]


def overhead(
    *, compute: Optional[Band], boundary: Optional[Corners], period_writes: Optional[Band],
    once: Optional[Band], k: Optional[int], x: float, passes: int,
) -> Optional[Dict[str, Any]]:
    """Toolchain-priced M at batch K beside the closed forms, per corner, with relative differences
    (``period_writes`` None: the program writes no weights per inference)."""
    if compute is None or boundary is None or once is None or not k or compute.low <= 0.0:
        return None
    c, o = corners(compute), corners(once)
    w = (0.0, 0.0, 0.0) if period_writes is None else corners(period_writes)
    kappa = [b / ci for b, ci in zip(boundary, c)]
    pi = [oi / ci for oi, ci in zip(o, c)]
    priced = [1.0 + ka + wi / k / ci for ka, wi, ci in zip(kappa, w, c)]
    stateless = [1.0 + p / k + ka for p, ka in zip(pi, kappa)]
    floor_phi = phi(x, passes)
    floor = None if floor_phi is None else [1.0 + floor_phi * p / k + ka for p, ka in zip(pi, kappa)]
    return {
        "corners": CORNER_NOTE, "kappa": kappa, "pi_corner_aligned": pi, "toolchain_priced": priced,
        "closed_form_stateless": stateless, "rel_diff_stateless": _rel(priced, stateless),
        "phi_floor": floor_phi, "closed_form_floor": floor,
        "rel_diff_floor": None if floor is None else _rel(priced, floor),
    }


def write_law_absence(terms: Mapping[str, Any]) -> str:
    """Why a row prints no write law at its batch: 'no physics declared' only where no pricing ran (``terms`` empty,
    no profile), else the batch the write law needs."""
    return "no inference_batch_K declared" if terms else NO_PHYSICS


def unlike_legs(legs: Optional[Mapping[str, Tuple[str, ...]]], compute_term: Any) -> Optional[str]:
    """[V-M1] Why a ratio over ``compute_term``'s cost is not printed, None where it may be: a cost that carries a
    static leg (a tuple of terms) is divided only by and into costs whose legs hold the same parts of the chip's static
    power (``legs``: by column, ``host_split.static_legs``); a caller that states no legs for such a cost is refused."""
    if isinstance(compute_term, str):
        return None
    if legs is None:
        raise ValueError(f"a ratio over {compute_term} needs each cost's static legs stated (host_split.static_legs)")
    held = {column: legs.get(column) for column in ("once", "period", "compute_only")}
    if len(set(held.values())) == 1:
        return None
    return "the costs hold unlike static legs: " + "; ".join(
        f"{column} {list(parts) if parts is not None else 'unstated'}" for column, parts in held.items())


def axis_columns(
    terms: Mapping[str, Mapping[str, Band]], refusals: Mapping[str, Mapping[str, str]],
    names: Sequence[Any], *, k: Optional[int], reading: RegimeReading, multiplexed: bool, x: float,
    passes: int, axis: str, legs: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> Dict[str, Any]:
    """One axis of a row from the pricings (program, compute_only, no_host) and the write law's tile writes (once,
    period).

    ``names``: (compute-only term, headline whose refusal names an absence, programming term, host term).
    M is priced for every program of two or more passes; its period writes are the
    write law's on the program's period census when the schedule writes per inference
    (``multiplexed``) and none otherwise. An axis is absent when its compute is refused, or
    [consts] when the write law refuses writing every tile once (by the law's reason).
    [U-SH] The compute-only and boundary terms are the one share rule's (``host_split.shares``); a boundary absent
    while the compute-only term prices is named (``boundary_absent``). [V-M1] A cost that carries a static leg is
    divided only over like legs (``unlike_legs`` on ``legs``): otherwise Pi, Pi with the host and M are absent, the
    two legs named.
    """
    compute_term, headline, programming, host_term = names
    t = {name: terms.get(name, {}) for name in ("program", "once", "period")}
    split = shares(terms, refusals, compute_term, headline)  # [U-SH] the one share rule of every plane
    compute, boundary, column = split["compute_only"], split["boundary"], share_column(split)
    once, period, host = t["once"].get(programming), t["period"].get(programming), t["program"].get(host_term)
    unlike = unlike_legs(legs, compute_term)  # [V-M1] a ratio only over like legs
    divisor = None if unlike else compute
    pi = ratio(once, divisor, f"Pi_{axis}")
    with_host = None if host is None or divisor is None else ratio(
        once, add_bands([divisor, host], "compute + host"), f"Pi_{axis} host included")
    pi_k = per_batch(pi, k)
    return {
        "terms": {**column["terms"], "once": triple(once), "period": triple(period), "host": triple(host)},
        "absent": (column["absent"] if compute is None else
                   "the write law refuses writing every tile once: "
                   + (refusals.get("once", {}).get(programming) or "no write term is priced") if once is None else
                   unlike),
        **({"boundary_absent": column["boundary_absent"]} if "boundary_absent" in column else {}),
        "host_share": share(host, t["program"].get(headline)),
        "pi": triple(pi), "pi_host_included": triple(with_host), "pi_over_k": triple(pi_k),
        "regime": regime(pi_k, reading, multiplexed),
        "overhead": overhead(compute=divisor, boundary=boundary, period_writes=period if multiplexed else None,
                             once=once, k=k, x=x, passes=passes) if passes >= 2 else None,
    }


PIPELINED = ("layer-pipelined lower bound on the time per inference of a resident program (no pass rewrites cores): "
             "one inference per stage time, a stage at least one latency step (compute-only time / latency_steps)")
#: [U-C2 G-2] On a chip priced per synaptic operation the compute-only time is one core's, which runs every event of
#: every stage in turn: a layer pipeline overlaps no stage, so the bound is the compute-only time itself.
PIPELINED_ONE_CORE = ("layer-pipelined lower bound on the time per inference of a resident program: one core runs "
                      "every synaptic operation of every stage in turn (t_synaptic_op), so stages cannot overlap and "
                      "the bound is the compute-only time")
#: [targeted M2] On several event-serial cores a layer pipeline overlaps the cores, never a core with itself.
PIPELINED_EVENT_CORES = ("layer-pipelined lower bound on the time per inference of a resident program on several "
                         "event-serial cores: each core runs its own synaptic operations one at a time "
                         "(t_synaptic_op), so no stage is shorter than the busiest core's operations of one pass "
                         "(core init and inter-core delivery not counted)")


def layer_pipelined(compute_only: Optional[Sequence[float]], latency_steps: Optional[int], *, multiplexed: bool,
                    passes: int, compute_law: Optional[str] = None,
                    event_stage: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The layer-pipelined bound of a resident program (``latency_semantics``): the compute-only time per latency
    step under the step law, the whole compute-only time on one event-serial core, [targeted M2] the busiest core's
    operations of one pass on several (``event_stage``); absent by name otherwise."""
    if not multiplexed and compute_law == EVENT_LAW and event_stage is not None:
        return {"layer_pipelined_bound_s": list(event_stage), "layer_pipelined_label": PIPELINED_EVENT_CORES}
    if not multiplexed and compute_only is not None and latency_steps:
        one_core = compute_law == EVENT_LAW
        bound = list(compute_only) if one_core else [v / latency_steps for v in compute_only]
        label = PIPELINED_ONE_CORE if one_core else PIPELINED
        return {"layer_pipelined_bound_s": bound, "layer_pipelined_label": label}
    return {"layer_pipelined_bound_s": None,
            "layer_pipelined_absent": (f"multiplexed: {passes} passes rewrite cores in sequence" if multiplexed
                                       else "the compute-only time or latency_steps is absent")}


def k_star(
    *, x_fluid: float, x_packed: float, max_writes: Optional[float], max_writes_source: str,
    rate_hz: Optional[float], lifetime_hours: Optional[float], endurance: Optional[float],
    pulses_per_write: Optional[float] = None,
) -> Dict[str, Any]:
    """K* in closed form and from the census, and the exact two-factor split of their ratio; [U-C C-1] the wear of a
    write is ``pulses_per_write`` endurance cycles where the technology declares it (printed beside)."""
    out: Dict[str, Any] = {
        "max_writes_per_cell_per_inference": max_writes, "max_writes_source": max_writes_source,
        "inference_rate_hz": rate_hz, "min_lifetime_hours": lifetime_hours, "endurance_cycles": endurance,
        **({} if pulses_per_write is None else {"pulses_per_write": pulses_per_write}),
    }
    if x_fluid > 1.0 and x_packed > 1.0 and max_writes is not None:
        packing = (x_packed - 1.0) / (x_fluid - 1.0)
        rotation = max_writes / (x_packed - 1.0)
        out.update(ratio=max_writes / (x_fluid - 1.0), factor_packing_replication=packing,
                   factor_rotation=rotation)
    else:
        out["ratio_absent"] = "x_fluid and x_packed must exceed 1 and the census must exist"
    missing = [n for n, v in (("inference_rate_hz", rate_hz), ("min_lifetime_hours", lifetime_hours),
                              ("endurance_cycles", endurance)) if v is None]
    if missing:
        out["k_star_absent"] = "undeclared: " + ", ".join(missing)
        return out
    assert rate_hz is not None and lifetime_hours is not None and endurance is not None
    service = rate_hz * lifetime_hours * _SECONDS_PER_HOUR / endurance
    if pulses_per_write is not None:
        service *= pulses_per_write
    out["k_star_closed"] = (x_fluid - 1.0) * service if x_fluid > 1.0 else None
    out["k_star_census"] = None if max_writes is None else max_writes * service
    return out
