#!/usr/bin/env python3
"""The ODIN power estimator: pilot gate and idle-subtracted power (H0, H4).

INPUTS. The sidecar sampler's readings (``odin_power_sampler.py``) and the
power protocol's markers (``odin_power_protocol.py``), both stamped on the
host's CLOCK_MONOTONIC, and the session plan whose ``thresholds`` were declared
before any reading was taken. Nothing here talks to a card; it runs the same
on the board node and at home on the archived files.

THE PILOT GATE (H1). Over the pilot window (stock xclbin loaded, kernel idle):
availability (readings with a power value over all reads), refresh (median
time between value changes), quantization (smallest step between distinct
readings), noise (standard deviation of distinct readings) and drift. Three
verdicts, declared in the plan:

  * NO_GO: no route returned a power reading, availability is under its
    floor, the reading never changed, or the refresh is slower than its
    ceiling. The protocol steps are skipped; the card power is reported as
    not measurable in the allocation.
  * GO: for every design of the full protocol, the predicted 95% half-width of
    the workload increment at the planned sample count (Student t over the
    planned rounds times the noise of one independent reading times
    sqrt(2 / readings per condition)) is within the declared fraction of that
    design's expected differential (the kernel's Vivado share, the gate's
    sizing value), the polled sample count after the warm-up meets the
    per-condition floor, and the quantum is resolvable (small against the
    expected differential, or dithered by noise at least as large as itself).
  * REDUCED: readings usable, but GO's prediction fails. The plan's reduced
    steps run (longer blocks on fewer designs); a bound is the expected result.

The per-condition counts the gate predicts with are taken AFTER the warm-up of
every window. The warm-up is the larger of 5 readings and 2 x the refresh the
pilot measured (``warmup_readings``), so the first readings of a window never
still show the previous condition; the value used is written into pilot.json
and estimate.json.

WHAT THE CARD BOUNDARY HOLDS. The sensor reads the whole card, so no difference
of two windows is the kernel alone. The declared quantities (``QUANTITIES``):

  * loop minus kernel idle is the WORKLOAD INCREMENT at the card boundary
    (kernel plus its HBM, interconnect and PCIe activity), an upper bound on
    the kernel's dynamic power, at kernel duty D. The loop drives the memory
    subsystem and the AXI interconnect (program, stimulus and capture traffic;
    the as-run arm re-reads the whole program every pass) and the shell's PCIe
    buffer syncs, none of which happen while the kernel idles;
  * the transfer control minus kernel idle (``transfer_<arm>``: the arm's
    buffer writes and syncs per sample, paced as in the loop, never
    ``ap_start``) is the TRANSFER-CONTROL INCREMENT, and loop minus the control
    of the same round is the workload increment NET OF THE TRANSFER CONTROL,
    the tighter upper bound on the kernel's dynamic power;
  * kernel idle minus the shell-only window is the USER-REGION INCREMENT over
    the shell (xclbin loaded and idle, minus shell after reset, valid only when
    the reset read back a zero UUID): everything the xclbin adds to the user
    region (HBM subsystem, clocking, interconnect, kernel), never the kernel's
    static power. It is refused unless the session read back a zero UUID.

A Vivado comparison, when the plan declares the routed report's numbers, sets
``odin_0`` at the arm's duty against the tightest measured upper bound, with
``hmss_0`` beside it; it is never a quotient, and it is refused for a design
whose xclbin was rebuilt after the archived report.

THE ESTIMATE (H3 to H4). Per design, rounds of interleaved windows (rotated
condition order) give paired differences d_r = mean(a_r) - mean(b_r) for
every declared contrast (a loop or control against kernel idle, a loop against
its control). The point is their mean. The interval is the WIDER of two 95%
intervals: a two-level block bootstrap (rounds resampled, and inside each
window moving blocks of readings at least one sensor refresh long), which is
conservative when within-window noise dominates, and a Student t interval over
the rounds (R - 1 degrees of freedom), which stays exact when between-round
drift dominates; the p-value is the larger of the two. Energy per inference per
round is d_r x wall_r / inferences_r. The user-region increment is not
interleaved, and is labelled so. Holm's step-down over the whole family (every
increment contrast) decides separation. Each contrast gets exactly one of
three labels:

  * NUMBER: separated, and the half-width is at most the declared fraction of
    the point (10%, VR section 5.5);
  * BOUND: separated, the half-width larger than that (or the end-of-session
    drift reference outside its band): the interval is printed, and its upper
    end is the bound;
  * NOT_SEPARATED: not separated from zero after Holm; the upper end of the
    interval is printed as the bound.

A window whose certificates or readouts failed is not used: the energy of a run
that did not compute the network is not the network's energy.

N COMPUTE UNITS (route D). A design the plan declares with ``compute_units`` N
> 1 ran N units of the kernel at once in every loop and control window. Its
quantities say so (D is then the mean kernel duty per unit, and inferences are
counted over all units), and every power contrast carries a per-unit figure,
the increment divided by N, named so and labelled with its parent's outcome.

THE SENSOR'S STEP (route D). When the plan declares ``thresholds.estimate.
resolution`` (``step_w``, the card's measured power step, and
``number_min_steps``), a NUMBER also needs its point at least that many steps
from zero, and no interval end below one step is offered as a bound: such an end
says the telemetry could not resolve the contrast, not that the kernel draws
less. Energy rows take the step per inference (step x window wall /
inferences). Without the declaration every label is the one above. With it,
the absolute drift band's quantum term is also floored at the step.

THE DRIFT REFERENCE OF AN INTERLEAVED CONTRAST (route D, fix round). When the
plan declares ``thresholds.estimate.drift`` with ``interleaved: idle_trend``,
an interleaved contrast (a loop or a control against kernel idle, a loop
against its control, and their energies) is judged by its own design's idle
trend instead of the session-end idle: the least-squares slope of that design's
kernel-idle window means against their mean reading times, times the mean over
the contrast's rounds of the time from the subtrahend window to the minuend
window, is the shift the trend puts on the contrast (energy rows scale each
round by its wall over inferences). The contrast's drift is inside its band when
that shift, added to the 95% half-width, stays within the declared fraction of
the point (the NUMBER rule's 10%): the trend is booked as an uncorrected error
of the point. The absolute gate (session-end idle against the pilot) stays the
drift reference of the one contrast that is not interleaved, the user-region
increment. Without the declaration the absolute gate judges every contrast.

    python3 host/odin_power_estimate.py pilot --samples S --markers M --plan P
    python3 host/odin_power_estimate.py estimate --samples S --markers M --plan P
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PILOT_SCHEMA = "odin_power_pilot/1"
ESTIMATE_SCHEMA = "odin_power_estimate/1"

GO, REDUCED, NO_GO = "GO", "REDUCED", "NO_GO"
VERDICTS = (GO, REDUCED, NO_GO)
NUMBER, BOUND, NOT_SEPARATED = "NUMBER", "BOUND", "NOT_SEPARATED"
OUTCOMES = (NUMBER, BOUND, NOT_SEPARATED)


#: Defaults for every declared threshold; a plan overrides any of them, and the
#: values actually used are written into every output.
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "pilot": {
        "availability_min": 0.95,
        "refresh_max_s": 5.0,
        "relative_half_width_max": 0.10,
        "samples_per_condition_min": 300,
        "quantum_max_fraction": 0.5,
        "changes_min": 3,
        # ESTIMATE: Vivado vectorless odin_0 power x the archived kernel duty
        # (scout_estimates_stdout.txt, ESTIMATE_duty_adjusted_differential_W).
        "expected_differential_w": {"stock": 0.0626, "wide": 0.0652},
    },
    "estimate": {
        "bootstrap": 2000,
        "seed": 20261023,
        "alpha": 0.05,
        # The warm-up of every window: the larger of this many readings and
        # ``warmup_refresh_multiple`` x the pilot's measured refresh.
        "warmup_samples": 5,
        "warmup_refresh_multiple": 2.0,
        "relative_half_width_max": 0.10,
        "drift_band_sd_multiple": 3.0,
        "drift_band_se_multiple": 3.0,
        "drift_window_s": 60.0,
        "rounds_min": 3,
    },
}

#: The declared quantities, word for word. ``D`` is replaced by the arm's
#: measured kernel duty and ``{arm}`` by the arm a control mirrors. No other
#: name is given to a difference of card readings.
QUANTITIES: Dict[str, str] = {
    "workload_increment_w": (
        "workload increment at the card boundary (kernel plus its HBM, "
        "interconnect and PCIe activity), an upper bound on the kernel's "
        "dynamic power, at kernel duty D"),
    "workload_energy_per_inference_mJ": (
        "workload energy per inference at the card boundary (the workload "
        "increment x window wall / inferences), an upper bound on the kernel's "
        "dynamic energy per inference, at kernel duty D"),
    "transfer_increment_w": (
        "transfer-control increment at the card boundary (the PCIe and HBM "
        "activity of the {arm} arm's buffer writes and syncs per sample, paced "
        "as in the loop, without ap_start)"),
    "transfer_energy_per_inference_mJ": (
        "transfer-control energy per inference at the card boundary (the "
        "transfer-control increment x window wall / replayed inferences)"),
    "net_of_transfers_w": (
        "workload increment net of the transfer control (kernel plus its own "
        "HBM and interconnect activity), an upper bound on the kernel's dynamic "
        "power, at kernel duty D"),
    "net_of_transfers_energy_per_inference_mJ": (
        "workload energy per inference net of the transfer control, an upper "
        "bound on the kernel's dynamic energy per inference, at kernel duty D"),
    "user_region_increment_w": (
        "user-region increment over the shell (xclbin loaded and idle, minus "
        "shell after reset, valid only when the reset read back a zero UUID)"),
}

UNITS = {"w": "W", "mJ": "mJ"}


def quantity(key: str, *, duty: Optional[float] = None, arm: Optional[str] = None,
             units: int = 1) -> str:
    """The declared label of ``key`` with the measured duty and the arm filled in.

    ``units`` > 1 (route D) appends what the card boundary then holds: N units
    running the same campaign at once.
    """
    text = QUANTITIES[key]
    has_duty = "duty D" in text
    if has_duty:
        text = text.replace("duty D", f"duty D = {duty:.3f}" if duty is not None
                            else "duty D (not recorded)")
    text = text.replace("{arm}", arm or "?")
    if int(units) > 1:
        text += (f"; {int(units)} compute units' buffer traffic replayed at once"
                 if key.startswith("transfer_") else
                 f"; {int(units)} compute units running the same campaign at once")
        text += ((", D the mean kernel duty per unit (each unit's run walls over its "
                  "own loop wall)" if has_duty else "")
                 + (", inferences counted over all units" if "per_inference" in key else ""))
    return text


def unit_of(key: str) -> str:
    return UNITS[key.rsplit("_", 1)[-1]]


def thresholds_of(plan: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    declared = plan.get("thresholds", {})
    for section in ("pilot", "estimate"):
        merged[section].update(declared.get(section, {}))
    return merged


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a torn last line from a killed job
    return rows


def windows_of(markers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every closed window: its start marker merged with its end marker."""
    open_windows: Dict[str, Dict[str, Any]] = {}
    closed: List[Dict[str, Any]] = []
    for record in markers:
        event = record.get("event")
        key = record.get("window_id")
        if event == "window_start" and key:
            open_windows[key] = dict(record)
        elif event == "window_end" and key in open_windows:
            window = open_windows.pop(key)
            window.update({k: v for k, v in record.items()
                           if k not in ("event", "t_mono", "state", "utc")})
            window["t_start"] = window.pop("t_mono")
            window["t_end"] = float(record["t_mono"])
            closed.append(window)
    return closed


def readings_in(rows: Sequence[Dict[str, Any]], t0: float, t1: float, *,
                warmup: int = 0) -> Tuple[List[Tuple[float, float]], int]:
    """((t, watts) of good readings inside [t0, t1], after warm-up), all reads."""
    inside = [row for row in rows if not row.get("header")
              and t0 <= float(row.get("t_mono", -1.0)) <= t1]
    inside.sort(key=lambda row: float(row["t_mono"]))
    good = [(float(row["t_mono"]), float(row["power_w"]))
            for row in inside if row.get("ok") and row.get("power_w") is not None]
    return good[int(warmup):], len(inside)


def temps_in(rows: Sequence[Dict[str, Any]], t0: float, t1: float) -> Dict[str, Dict[str, float]]:
    per: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("ok") and t0 <= float(row.get("t_mono", -1.0)) <= t1:
            for name, value in (row.get("temps_c") or {}).items():
                per.setdefault(name, []).append(float(value))
    return {name: {"mean": statistics.mean(v), "min": min(v), "max": max(v)}
            for name, v in sorted(per.items()) if v}


def sampler_interval(headers: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> float:
    stamps = sorted(float(r["t_mono"]) for r in rows if not r.get("header") and "t_mono" in r)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if gaps:
        return statistics.median(gaps)
    for header in headers:
        if header.get("interval_s"):
            return float(header["interval_s"])
    return 1.0


# ---------------------------------------------------------------------------
# The pilot
# ---------------------------------------------------------------------------


def characterize(readings: Sequence[Tuple[float, float]], reads: int, *,
                 window_s: float) -> Dict[str, Any]:
    """Availability, refresh, quantum, noise and drift of one idle window."""
    values = [watts for _t, watts in readings]
    out: Dict[str, Any] = {
        "reads": int(reads), "readings": len(values),
        "availability": (len(values) / reads) if reads else 0.0,
    }
    if not values:
        out.update({"changes": 0, "refresh_s": None, "quantum_w": None,
                    "sigma_w": None, "mean_w": None})
        return out
    change_times: List[float] = []
    distinct: List[float] = [values[0]]
    steps: List[float] = []
    for (t_prev, w_prev), (t_now, w_now) in zip(readings, readings[1:]):
        if w_now != w_prev:
            change_times.append(t_now)
            distinct.append(w_now)
            steps.append(abs(w_now - w_prev))
    gaps = [b - a for a, b in zip(change_times, change_times[1:])]
    out["changes"] = len(change_times)
    out["refresh_s"] = statistics.median(gaps) if gaps else None
    out["quantum_w"] = round(min(steps), 9) if steps else None
    out["sigma_w"] = statistics.stdev(distinct) if len(distinct) >= 2 else 0.0
    out["sigma_all_w"] = statistics.stdev(values) if len(values) >= 2 else 0.0
    out["mean_w"] = statistics.mean(values)
    out["distinct_readings"] = len(distinct)
    # Drift: slope of a least-squares line, and the spread of window means.
    times = [t for t, _w in readings]
    t_mean = statistics.mean(times)
    denom = sum((t - t_mean) ** 2 for t in times)
    out["drift_w_per_100s"] = (
        100.0 * sum((t - t_mean) * (w - out["mean_w"]) for t, w in readings) / denom
        if denom > 0 else 0.0)
    means = []
    start = times[0]
    while start < times[-1]:
        chunk = [w for t, w in readings if start <= t < start + window_s]
        if len(chunk) >= 2:
            means.append(statistics.mean(chunk))
        start += window_s
    out["window_s"] = window_s
    out["window_means_w"] = means
    out["window_mean_sd_w"] = statistics.stdev(means) if len(means) >= 2 else None
    return out


def warmup_readings(gate: Dict[str, Any], *, interval_s: float,
                    refresh_s: Optional[float]) -> int:
    """Readings dropped at the start of every window: the larger of
    ``warmup_samples`` readings and ``warmup_refresh_multiple`` x the refresh."""
    floor = int(gate["warmup_samples"])
    if not refresh_s or interval_s <= 0:
        return floor
    multiple = float(gate.get("warmup_refresh_multiple", 2.0))
    # The 1e-9 keeps an exact multiple (2 x 1.0 s / 0.25 s = 8) from rounding up.
    return max(floor, int(math.ceil(multiple * float(refresh_s) / float(interval_s) - 1e-9)))


def warmup_record(gate: Dict[str, Any], *, interval_s: float,
                  refresh_s: Optional[float]) -> Dict[str, Any]:
    """The warm-up as pilot.json and estimate.json record it."""
    readings = warmup_readings(gate, interval_s=interval_s, refresh_s=refresh_s)
    return {
        "readings": readings, "seconds": readings * float(interval_s),
        "interval_s": float(interval_s), "refresh_s": refresh_s,
        "rule": (f"the larger of {int(gate['warmup_samples'])} readings and "
                 f"{float(gate.get('warmup_refresh_multiple', 2.0)):g} x the measured "
                 f"refresh, in readings at the sampler's interval, dropped at the "
                 f"start of every window"),
    }


def planned_counts(step: Dict[str, Any], interval_s: float, refresh_s: float,
                   warmup: int = 0) -> Tuple[float, float]:
    """(polled readings, independent readings) per condition of a protocol step,
    after ``warmup`` readings are dropped from each of its windows."""
    block = float(step.get("block_s", 60.0))
    usable = float(step.get("rounds", 5)) * max(0.0, block - int(warmup) * interval_s)
    polled = usable / max(interval_s, 1e-9)
    independent = usable / max(refresh_s, interval_s, 1e-9)
    return polled, independent


def pilot_verdict(summary: Dict[str, Any], plan: Dict[str, Any], *,
                  interval_s: float, probe_ok: bool = True,
                  warmup: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GO, REDUCED or NO_GO, with every number the decision used."""
    gate = thresholds_of(plan)["pilot"]
    if warmup is None:
        warmup = warmup_record(thresholds_of(plan)["estimate"], interval_s=interval_s,
                               refresh_s=summary.get("refresh_s"))
    reasons: List[str] = []
    out: Dict[str, Any] = {"thresholds": gate, "interval_s": interval_s,
                           "predictions": [], "warmup": warmup}
    if not probe_ok:
        reasons.append("no telemetry route returned a board power reading")
    if summary["availability"] < float(gate["availability_min"]):
        reasons.append(f"availability {summary['availability']:.3f} < "
                       f"{gate['availability_min']}")
    if summary.get("changes", 0) < int(gate["changes_min"]):
        reasons.append(f"the reading changed {summary.get('changes', 0)} time(s) "
                       f"in the pilot (< {gate['changes_min']}): no refresh to measure")
    refresh = summary.get("refresh_s")
    if refresh is not None and refresh > float(gate["refresh_max_s"]):
        reasons.append(f"refresh {refresh:.2f} s > {gate['refresh_max_s']} s")
    if reasons:
        out.update({"verdict": NO_GO, "reasons": reasons})
        return out
    refresh = float(refresh if refresh is not None else interval_s)
    sigma = float(summary.get("sigma_w") or 0.0)
    measured_quantum = float(summary.get("quantum_w") or 0.0)
    # A declared floor (route D: the card's measured current step) replaces a
    # smaller measured step, which may be a voltage step and not the resolution.
    quantum = max(measured_quantum, float(gate.get("quantum_floor_w") or 0.0))
    # A quantizer without dither cannot be averaged below its step; with no
    # noise the uniform-error floor q / sqrt(12) is the honest sigma.
    sigma_eff = max(sigma, quantum / math.sqrt(12.0))
    expected = gate["expected_differential_w"]

    def check(step: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        design = step["design"]
        polled, independent = planned_counts(step, interval_s, refresh,
                                             warmup=int(warmup["readings"]))
        # The critical value the estimate will apply: Student t over the step's
        # rounds (2.776 at 5 rounds), never the optimistic normal 1.96.
        critical = t_critical(max(1, int(step.get("rounds", 5)) - 1), 0.05)
        half = critical * sigma_eff * math.sqrt(2.0 / max(independent, 1.0))
        dp = float(expected.get(design, 0.0))
        row = {
            "step": step.get("label", step["kind"]), "design": design,
            "when": step.get("when"), "polled_per_condition": polled,
            "independent_per_condition": independent,
            "warmup_readings_per_window": int(warmup["readings"]),
            "sigma_eff_w": sigma_eff, "t_critical": critical,
            "predicted_half_width_w": half,
            "expected_differential_w": dp,
            "half_width_ok": dp > 0 and half <= float(gate["relative_half_width_max"]) * dp,
            "samples_ok": polled >= float(gate["samples_per_condition_min"]),
            "quantum_ok": (quantum <= float(gate["quantum_max_fraction"]) * dp
                           or sigma >= quantum),
        }
        if gate.get("quantum_floor_w") is not None:
            row.update({"quantum_used_w": quantum, "quantum_measured_w": measured_quantum,
                        "quantum_floor_w": float(gate["quantum_floor_w"])})
        row["ok"] = row["half_width_ok"] and row["samples_ok"] and row["quantum_ok"]
        return row["ok"], row

    go_steps = [s for s in plan["steps"] if s["kind"] == "protocol"
                and s.get("when", "go") in ("go", "go_or_reduced")]
    checks = [check(step) for step in go_steps]
    out["predictions"] = [row for _ok, row in checks]
    reduced_steps = [s for s in plan["steps"] if s["kind"] == "protocol"
                     and s.get("when") in ("reduced", "go_or_reduced")]
    out["predictions"] += [check(step)[1] for step in reduced_steps]
    if checks and all(ok for ok, _row in checks):
        out.update({"verdict": GO, "reasons": [
            "every full-protocol design is predicted to meet the half-width rule"]})
    else:
        failing = [row for ok, row in checks if not ok]
        out.update({"verdict": REDUCED, "reasons": [
            f"{row['design']}: predicted half-width "
            f"{row['predicted_half_width_w'] * 1e3:.1f} mW against "
            f"{float(gate['relative_half_width_max']) * row['expected_differential_w'] * 1e3:.1f} mW"
            f" allowed; samples_ok={row['samples_ok']}, quantum_ok={row['quantum_ok']}"
            for row in failing] or ["no full-protocol step declared"]})
    return out


def pilot(samples_path: str, markers_path: str, plan: Dict[str, Any], *,
          probe_ok: bool = True) -> Dict[str, Any]:
    loaded = load_jsonl(samples_path)
    headers = [r for r in loaded if r.get("header")]
    rows = [r for r in loaded if not r.get("header")]
    windows = [w for w in windows_of(load_jsonl(markers_path)) if w.get("kind") == "pilot"]
    interval = sampler_interval(headers, rows)
    gate = thresholds_of(plan)
    result: Dict[str, Any] = {"schema": PILOT_SCHEMA, "interval_s": interval}
    if not windows:
        summary = {"reads": 0, "readings": 0, "availability": 0.0, "changes": 0}
        result.update({"summary": summary, **pilot_verdict(
            summary, plan, interval_s=interval, probe_ok=False)})
        result["reasons"] = ["no pilot window was recorded"] + result["reasons"]
        return result
    window = windows[-1]
    window_s = float(gate["estimate"]["drift_window_s"])
    # Two passes: the refresh is measured with the floor warm-up, then the
    # declared warm-up (which depends on it) is applied and the pilot is read
    # again with it.
    first, reads = readings_in(rows, window["t_start"], window["t_end"],
                               warmup=int(gate["estimate"]["warmup_samples"]))
    warmup = warmup_record(gate["estimate"], interval_s=interval,
                           refresh_s=characterize(first, reads, window_s=window_s)
                           .get("refresh_s"))
    readings, reads = readings_in(rows, window["t_start"], window["t_end"],
                                  warmup=int(warmup["readings"]))
    summary = characterize(readings, reads, window_s=window_s)
    # Availability counts every good reading over every read; the warm-up is a
    # choice of the estimator, not a missing reading.
    every, _ = readings_in(rows, window["t_start"], window["t_end"], warmup=0)
    summary["availability"] = (len(every) / reads) if reads else 0.0
    summary["temps_c"] = temps_in(rows, window["t_start"], window["t_end"])
    summary["window"] = {k: window.get(k) for k in ("design", "t_start", "t_end")}
    summary["latency_s_median"] = statistics.median(
        [float(r["latency_s"]) for r in rows
         if window["t_start"] <= float(r.get("t_mono", -1)) <= window["t_end"]] or [0.0])
    result["summary"] = summary
    result.update(pilot_verdict(summary, plan, interval_s=interval, probe_ok=probe_ok,
                                warmup=warmup))
    statuses = xclbin_statuses(load_jsonl(markers_path)).get(str(window.get("design")))
    declared = (plan.get("designs") or {}).get(str(window.get("design"))) or {}
    rebuilt = statuses == "rebuilt" or bool(
        (declared.get("xclbin_identity") or {}).get("rebuilt"))
    re_extracted = bool((declared.get("vivado") or {}).get("re_extracted"))
    archived_text = ("the archived builds' Vivado report_power (odin_0 x archived duty, "
                     "ESTIMATE)")
    result["expected_differential_describes"] = (
        str(gate["pilot"].get("expected_differential_class") or archived_text)
        if rebuilt else archived_text)
    if rebuilt and re_extracted:
        result["notes"] = [
            "the pilot design's xclbin was rebuilt; its expected differential comes "
            "from that build's own re-extracted report_power (vivado.re_extracted)"]
    elif rebuilt:
        result["notes"] = [
            "the pilot design's xclbin was rebuilt: the expected differential comes "
            "from the archived report_power, which does not describe this build; "
            "re-extract report_power from the new build before quoting it"]
    return result


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------


class _Series:
    """One window's readings with O(1) block means for the moving-block bootstrap."""

    def __init__(self, values: Sequence[float], block: int) -> None:
        self.values = list(values)
        self.n = len(self.values)
        self.block = max(1, min(int(block), self.n)) if self.n else 1
        self.prefix = [0.0]
        for value in self.values:
            self.prefix.append(self.prefix[-1] + value)
        self.mean = (self.prefix[-1] / self.n) if self.n else float("nan")

    def resample_mean(self, rng: random.Random) -> float:
        if self.n == 0:
            return float("nan")
        span = self.n - self.block + 1
        draws = int(math.ceil(self.n / float(self.block)))
        total = 0.0
        for _ in range(draws):
            start = rng.randrange(span)
            total += self.prefix[start + self.block] - self.prefix[start]
        return total / (draws * self.block)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (Lentz's method)."""
    tiny, eps = 1e-300, 3e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with ``df`` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def t_critical(df: float, alpha: float = 0.05) -> float:
    """The two-sided critical value: t with P(|T| >= t) = alpha."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bootstrap_contrast(rounds: Sequence[Tuple[_Series, _Series, float]], *,
                       draws: int, rng: random.Random) -> Dict[str, Any]:
    """Paired rounds (a, b, scale): point and CI of mean_r scale_r x (a_r - b_r).

    The declared interval is the wider of the two-level block bootstrap and the
    t interval over rounds; the declared p-value is the larger of theirs.
    """
    diffs = [scale * (a.mean - b.mean) for a, b, scale in rounds]
    point = statistics.mean(diffs)
    samples: List[float] = []
    count = len(rounds)
    for _ in range(int(draws)):
        total = 0.0
        for _k in range(count):
            a, b, scale = rounds[rng.randrange(count)]
            total += scale * (a.resample_mean(rng) - b.resample_mean(rng))
        samples.append(total / count)
    samples.sort()
    boot_lo, boot_hi = _percentile(samples, 0.025), _percentile(samples, 0.975)
    below = sum(1 for value in samples if value <= 0.0)
    above = sum(1 for value in samples if value >= 0.0)
    boot_p = min(1.0, 2.0 * (min(below, above) + 1) / (len(samples) + 1))
    # The t interval over rounds. A ``b`` series shared by every round (the one
    # shell-only window under the user-region increment) adds its own mean's variance.
    variance = statistics.variance(diffs) / count if count >= 2 else float("inf")
    shared = all(b is rounds[0][1] for _a, b, _s in rounds)
    if shared and count >= 2:
        b = rounds[0][1]
        scale = statistics.mean(s for _a, _b, s in rounds)
        if b.n >= 2:
            effective = max(1.0, b.n / float(b.block))
            variance += (scale ** 2) * statistics.variance(b.values) / effective
    se = math.sqrt(variance) if variance > 0 else 0.0
    df = max(1, count - 1)
    crit = t_critical(df, 0.05)
    t_lo, t_hi = point - crit * se, point + crit * se
    t_p = (t_two_sided_p(point / se, df) if se > 0
           else (0.0 if point != 0 else 1.0))
    lo, hi = min(boot_lo, t_lo), max(boot_hi, t_hi)
    return {"point": point, "lo": lo, "hi": hi, "half_width": 0.5 * (hi - lo),
            "p_value": max(boot_p, t_p), "rounds": count, "draws": int(draws),
            "round_differences": diffs,
            "bootstrap": {"lo": boot_lo, "hi": boot_hi, "p_value": boot_p},
            "t_interval": {"lo": t_lo, "hi": t_hi, "p_value": t_p, "df": df,
                           "critical": crit, "se": se}}


def holm(p_values: Dict[str, float], alpha: float) -> Dict[str, bool]:
    """Holm's step-down: which hypotheses are rejected at family-wise alpha."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    rejected: Dict[str, bool] = {name: False for name in p_values}
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        if p_value <= alpha / (total - rank):
            rejected[name] = True
        else:
            break
    return rejected


def label(contrast: Dict[str, Any], *, separated: bool, relative_max: float,
          drift_ok: bool = True, unit: str = "W",
          name: Optional[str] = None,
          resolution: Optional[Dict[str, float]] = None,
          drift_reason: Optional[str] = None) -> Tuple[str, str]:
    """(outcome, the sentence the record carries) for one contrast.

    The sentence opens with the contrast's declared quantity (``name``, or the
    contrast's own ``quantity``), so no number travels without what it is.
    ``resolution`` ({"step": the sensor's step in ``unit``, "min_steps": k})
    adds the step rules; without it the labels are the archived ones.
    ``drift_reason`` (route D's declared drift rule) names why the drift
    reference failed, and every failed rule is then named; without it a drift
    failure hides the half-width reason, as in the archived labels.
    """
    head = name or contrast.get("quantity")
    lead = f"{head}: " if head else ""
    step = None if not resolution else float(resolution["step"])
    below_step = step is not None and contrast["hi"] < step
    if not separated:
        if below_step:
            return NOT_SEPARATED, (
                f"{lead}not separated from zero after Holm; its 95% upper end "
                f"({contrast['hi']:.6g} {unit}) lies below one sensor step "
                f"({step:.6g} {unit}), so no upper bound is offered")
        return NOT_SEPARATED, (
            f"{lead}not separated from zero after Holm; at most {contrast['hi']:.6g} "
            f"{unit} (95% upper end)")
    point, half = contrast["point"], contrast["half_width"]
    min_steps = float(resolution["min_steps"]) if resolution else 0.0
    resolved = step is None or abs(point) >= min_steps * step
    if abs(point) > 0 and half <= relative_max * abs(point) and drift_ok and resolved:
        return NUMBER, f"{lead}{point:.6g} +/- {half:.2g} {unit} (95%)"
    reasons = []
    half_ok = abs(point) > 0 and half <= relative_max * abs(point)
    if drift_reason is None:
        if not drift_ok:
            reasons.append("drift reference outside its band")
        elif not half_ok:
            reasons.append(f"half-width {half:.2g} above {relative_max:.0%} of the point")
    else:
        if not half_ok:
            reasons.append(f"half-width {half:.2g} above {relative_max:.0%} of the point")
        if not drift_ok:
            reasons.append(drift_reason)
    if not resolved:
        reasons.append(f"the point is under {min_steps:g} sensor steps of "
                       f"{step:.6g} {unit}")
    text = (f"{lead}between {contrast['lo']:.6g} and {contrast['hi']:.6g} {unit} "
            f"(95%); {'; '.join(reasons)}")
    if below_step:
        text += "; its upper end lies below one sensor step, so it is not offered as a bound"
    return BOUND, text


def _series(rows: Sequence[Dict[str, Any]], window: Dict[str, Any], *,
            warmup: int, block: int) -> Tuple[_Series, int]:
    readings, reads = readings_in(rows, window["t_start"], window["t_end"], warmup=warmup)
    return _Series([watts for _t, watts in readings], block), reads


def _mean_time(rows: Sequence[Dict[str, Any]], window: Dict[str, Any], *,
               warmup: int) -> float:
    """The mean time of the readings a window's mean is taken over."""
    readings, _reads = readings_in(rows, window["t_start"], window["t_end"], warmup=warmup)
    if readings:
        return statistics.mean(t for t, _w in readings)
    return 0.5 * (float(window["t_start"]) + float(window["t_end"]))


#: The declared names of the drift rules an interleaved contrast may take.
DRIFT_RULES = ("idle_trend",)


def idle_trend(points: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Least-squares slope of idle-window means (t, W) against time, with its SE."""
    count = len(points)
    out: Dict[str, Any] = {"idle_windows": count, "slope_w_per_s": None,
                           "slope_se_w_per_s": None,
                           "points": [[t, w] for t, w in points]}
    if count < 2:
        return out
    t_mean = statistics.mean(t for t, _w in points)
    w_mean = statistics.mean(w for _t, w in points)
    sxx = sum((t - t_mean) ** 2 for t, _w in points)
    if sxx <= 0:
        return out
    slope = sum((t - t_mean) * (w - w_mean) for t, w in points) / sxx
    out["slope_w_per_s"] = slope
    if count >= 3:
        residual = sum((w - w_mean - slope * (t - t_mean)) ** 2 for t, w in points)
        out["slope_se_w_per_s"] = math.sqrt(residual / (count - 2) / sxx)
    return out


def _rounds(rows: Sequence[Dict[str, Any]], by_round: Dict[int, Dict[str, Dict[str, Any]]],
            minuend: str, subtrahend: str, *, warmup: int, block: int,
            names: Tuple[str, str]) -> Dict[str, Any]:
    """Every round holding both conditions: the paired series of one contrast.

    ``power`` pairs (a, b, 1) give mean_r(a_r - b_r); ``energy`` pairs carry the
    minuend window's wall over its inferences, in mJ per inference.
    """
    power: List[Tuple[_Series, _Series, float]] = []
    energy: List[Tuple[_Series, _Series, float]] = []
    card: List[float] = []
    duty: List[float] = []
    duty_window: List[float] = []
    power_offsets: List[float] = []
    energy_offsets: List[float] = []
    inferences = 0.0
    counts = {names[0]: 0, names[1]: 0}
    for _number, conds in sorted(by_round.items()):
        if minuend not in conds or subtrahend not in conds:
            continue
        a_w, b_w = conds[minuend], conds[subtrahend]
        a_s, _ = _series(rows, a_w, warmup=warmup, block=block)
        b_s, _ = _series(rows, b_w, warmup=warmup, block=block)
        if a_s.n == 0 or b_s.n == 0:
            continue
        counts[names[0]] += a_s.n
        counts[names[1]] += b_s.n
        wall = float(a_w["t_end"]) - float(a_w["t_start"])
        n_inf = float(a_w.get("inferences", 0) or 0)
        # The time from the subtrahend's readings to the minuend's: what a
        # trend of the baseline adds to this round's difference.
        offset = (_mean_time(rows, a_w, warmup=warmup)
                  - _mean_time(rows, b_w, warmup=warmup))
        power.append((a_s, b_s, 1.0))
        power_offsets.append(offset)
        if n_inf > 0:
            # W x s / inference = J per inference; reported in mJ.
            energy.append((a_s, b_s, 1e3 * wall / n_inf))
            energy_offsets.append(offset)
            card.append(1e3 * a_s.mean * wall / n_inf)
            inferences += n_inf
        if a_w.get("kernel_duty") is not None:
            duty.append(float(a_w["kernel_duty"]))
        if a_w.get("kernel_duty_window") is not None:
            duty_window.append(float(a_w["kernel_duty_window"]))
    return {"power": power, "energy": energy, "card": card, "duty": duty,
            "duty_window": duty_window, "power_offsets": power_offsets,
            "energy_offsets": energy_offsets,
            "inferences": inferences, "counts": counts}


def xclbin_statuses(markers: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Per design, the identity status its xclbin loads recorded ('rebuilt' wins)."""
    seen: Dict[str, List[str]] = {}
    for record in markers:
        if record.get("event") == "xclbin_loaded" and record.get("design"):
            seen.setdefault(str(record["design"]), []).append(
                str(record.get("xclbin_status") or "not recorded"))
    out: Dict[str, str] = {}
    for design, statuses in seen.items():
        out[design] = ("rebuilt" if "rebuilt" in statuses
                       else "archived" if "archived" in statuses else statuses[-1])
    return out


#: Tightest first: the MEASURED upper bounds on the kernel's dynamic power.
UPPER_BOUNDS = ("net_of_transfers_w", "workload_increment_w")


def vivado_comparison(record: Dict[str, Any], declared: Optional[Dict[str, Any]], *,
                      status: Optional[str], units: int = 1) -> Optional[Dict[str, Any]]:
    """``odin_0`` at duty against the tightest measured upper bound, ``hmss_0`` beside.

    Never a quotient: a card-boundary difference holds more than the kernel, so
    it can show that Vivado's kernel price fits under it, or that it does not,
    and nothing else. For an N-unit design (``units`` > 1) the declared
    ``odin_0_dynamic_w`` is the sum of the N ``odin_<k>`` rows, and the reading
    names that sum and the mean duty per unit.
    """
    if not declared:
        return None
    if status == "rebuilt" and not declared.get("re_extracted"):
        return {"refused": (
            "the xclbin was rebuilt after the archived build: the archived "
            "report_power does not describe this placement; re-extract report_power "
            "from the new build (and set vivado.re_extracted) before quoting odin_0, "
            "hmss_0 or the gate's expected differential")}
    odin, hmss = float(declared["odin_0_dynamic_w"]), float(declared["hmss_0_dynamic_w"])
    out: Dict[str, Any] = {
        "class": declared.get("class", "ESTIMATE"), "source": declared.get("source"),
        "describes": declared.get("describes"),
        "rule": ("Vivado's odin_0 dynamic power (ESTIMATE) at the arm's kernel duty, "
                 "set against the tightest MEASURED upper bound on the kernel's dynamic "
                 "power, with hmss_0 (the HBM subsystem, ESTIMATE) beside it; never "
                 "expressed as a quotient of the two"),
        "arms": {},
    }
    for arm, row in sorted(record["arms"].items()):
        key = next((k for k in UPPER_BOUNDS
                    if isinstance(row.get(k), dict) and "hi" in row[k]), None)
        if key is None:
            continue  # a refused arm holds no interval to set odin_0 against
        bound = row[key]
        duty = row.get("kernel_duty")
        at_duty = odin * duty if duty is not None else None
        within = None if at_duty is None else bool(at_duty <= bound["hi"])
        many = int(units) > 1
        per_unit = " per unit" if many else ""
        priced = (f"{at_duty * 1e3:.1f} mW at {'mean ' if many else ''}kernel duty "
                  f"{duty:.3f}{per_unit}"
                  if at_duty is not None else "at an unrecorded duty")
        verb = {True: "lies under", False: "exceeds", None: "cannot be set against"}[within]
        what = (f"the {int(units)} compute units (the sum of odin_0 to odin_{int(units) - 1})"
                if many else "odin_0")
        out["arms"][arm] = {
            "kernel_duty": duty, "odin_0_dynamic_w": odin, "odin_0_at_duty_w": at_duty,
            "hmss_0_dynamic_w": hmss, "upper_bound_w": bound["hi"],
            "upper_bound_quantity": key, "upper_bound_outcome": bound.get("outcome"),
            "odin_0_within_bound": within,
            "reading": (
                f"Vivado prices {what} at {odin * 1e3:.1f} mW dynamic, {priced} "
                f"(ESTIMATE); it {verb} the measured upper bound of "
                f"{bound['hi'] * 1e3:.1f} mW ({key}, 95% upper end, MEASURED); "
                f"hmss_0, the HBM subsystem, is priced at {hmss * 1e3:.0f} mW dynamic "
                f"beside it (ESTIMATE)"),
        }
        if many:
            out["arms"][arm]["compute_units"] = int(units)
    return out


def estimate(samples_path: str, markers_path: str, plan: Dict[str, Any], *,
             pilot_result: Optional[Dict[str, Any]] = None,
             shell_verified: Optional[bool] = None) -> Dict[str, Any]:
    gate = thresholds_of(plan)["estimate"]
    loaded = load_jsonl(samples_path)
    headers = [r for r in loaded if r.get("header")]
    rows = [r for r in loaded if not r.get("header")]
    markers = load_jsonl(markers_path)
    windows = windows_of(markers)
    if shell_verified is None:
        # The session's own record of the reset: True only when a zero UUID was
        # read back after it (None when it was skipped or never read back).
        resets = [m for m in markers if m.get("event") == "reset"]
        if resets:
            shell_verified = resets[-1].get("shell_verified")
    interval = sampler_interval(headers, rows)
    refresh = None
    if pilot_result and pilot_result.get("summary"):
        refresh = pilot_result["summary"].get("refresh_s")
    if pilot_result and pilot_result.get("warmup"):
        warm = dict(pilot_result["warmup"])
    else:
        warm = warmup_record(gate, interval_s=interval, refresh_s=refresh)
    warmup = int(warm["readings"])
    rng = random.Random(int(gate["seed"]))
    block_len = max(1, int(math.ceil((refresh or interval) / max(interval, 1e-9))))
    statuses = xclbin_statuses(markers)
    out: Dict[str, Any] = {
        "schema": ESTIMATE_SCHEMA, "thresholds": gate, "interval_s": interval,
        "refresh_s": refresh, "bootstrap_block_readings": block_len,
        "warmup": warm, "shell_verified": shell_verified,
        "quantities": dict(QUANTITIES),
        "designs": {}, "excluded_windows": [], "notes": [],
    }

    resolution_declared = gate.get("resolution") or None
    drift_rule = gate.get("drift") or None
    if drift_rule is not None and drift_rule.get("interleaved") not in DRIFT_RULES:
        raise ValueError(
            f"thresholds.estimate.drift.interleaved is {drift_rule.get('interleaved')!r}; "
            f"the declared rules are {DRIFT_RULES} (idle_trend: each design's own "
            f"kernel-idle trend, set against each interleaved contrast)")
    relative_max = float(gate["relative_half_width_max"])

    # Drift reference: the end-of-session idle of the pilot's design against the
    # pilot itself, judged against the pilot's own 60 s window-mean spread. With
    # route D's drift rule it judges only the contrast that is not interleaved.
    drift_ok = True
    pilots = [w for w in windows if w.get("kind") == "pilot"]
    drifts = [w for w in windows if w.get("kind") == "drift"]
    if pilots and drifts:
        p_read, _ = readings_in(rows, pilots[-1]["t_start"], pilots[-1]["t_end"], warmup=warmup)
        d_read, _ = readings_in(rows, drifts[-1]["t_start"], drifts[-1]["t_end"], warmup=warmup)
        if p_read and d_read:
            p_sum = characterize(p_read, len(p_read), window_s=float(gate["drift_window_s"]))
            difference = statistics.mean(w for _t, w in d_read) - p_sum["mean_w"]
            spread = p_sum.get("window_mean_sd_w") or 0.0

            def se2(values: List[float]) -> float:
                # The variance of a window mean, with one effective reading per
                # sensor refresh (the bootstrap's block length).
                if len(values) < 2:
                    return 0.0
                return statistics.variance(values) / max(1.0, len(values) / float(block_len))

            noise = math.sqrt(se2([w for _t, w in p_read]) + se2([w for _t, w in d_read]))
            # Inside the band: within the pilot's own 60 s window-mean spread, or
            # within the two means' measurement noise, or within one quantum
            # (with a declared sensor step, never less than that step).
            measured_quantum = float(p_sum.get("quantum_w") or 0.0)
            quantum = (max(measured_quantum, float(resolution_declared["step_w"]))
                       if resolution_declared else measured_quantum)
            band = max(float(gate["drift_band_sd_multiple"]) * spread,
                       float(gate["drift_band_se_multiple"]) * noise,
                       quantum)
            drift_ok = abs(difference) <= band
            out["drift"] = {"design": drifts[-1].get("design"),
                            "difference_w": difference, "band_w": band,
                            "window_mean_sd_w": spread, "noise_se_w": noise,
                            "inside_band": drift_ok}
            if resolution_declared:
                out["drift"].update({"quantum_w": quantum,
                                     "quantum_measured_w": measured_quantum})
            if drift_rule is not None:
                out["drift"].update({
                    "applies_to": ["user_region_increment_w"],
                    "rule": ("the session-end idle of the pilot's design against the "
                             "pilot; the drift reference of the contrast that is not "
                             "interleaved (every interleaved contrast is judged by its "
                             "own design's idle trend)")})
    else:
        out["notes"].append("no drift reference window; drift not checked")

    shells = [w for w in windows if w.get("kind") == "shell_only"]
    contrasts: Dict[str, Dict[str, Any]] = {}
    #: energy row -> the power contrast whose separation it inherits (same rounds).
    energies: List[Tuple[Dict[str, Any], str]] = []
    rebuilt: Dict[str, bool] = {}
    protocol = [w for w in windows if w.get("kind") == "protocol"]
    designs_declared = plan.get("designs", {}) or {}
    draws = int(gate["bootstrap"])
    trends: Dict[str, Dict[str, Any]] = {}

    def units_of(design: str) -> int:
        return int((designs_declared.get(design) or {}).get("compute_units", 1))

    def trend_shift(row: Dict[str, Any], design: str, offsets: Sequence[float],
                    scales: Sequence[float], unit: str) -> None:
        """Route D's drift rule on one interleaved row: the design's idle trend
        at the row's own window times, booked against the 10% rule's room."""
        trend = trends.get(design) or {}
        slope, slope_se = trend.get("slope_w_per_s"), trend.get("slope_se_w_per_s")
        record: Dict[str, Any] = {
            "rule": ("the design's idle trend (least-squares slope of its kernel-idle "
                     "window means) x the mean, over the contrast's rounds, of the time "
                     "from the subtrahend window to the minuend window; inside the band "
                     "when the shift plus the 95% half-width is at most "
                     f"{relative_max:.0%} of the point"),
            "unit": unit, "slope_w_per_s": slope, "slope_se_w_per_s": slope_se,
            "idle_windows": trend.get("idle_windows", 0),
            "mean_offset_s": statistics.mean(offsets) if offsets else None,
        }
        if slope is None or not offsets:
            record.update({"shift": None, "band": None, "inside_band": False,
                           "why": "fewer than two idle windows: no idle trend"})
        else:
            weighted = statistics.mean(scale * offset
                                       for scale, offset in zip(scales, offsets))
            shift = slope * weighted
            band = max(0.0, relative_max * abs(row["point"]) - row["half_width"])
            record.update({
                "shift": shift, "band": band,
                "shift_se": (abs(weighted) * slope_se) if slope_se is not None else None,
                "inside_band": abs(shift) <= band})
        row["drift_trend"] = record

    def contrast(paired: Dict[str, Any], key: str, family_key: str, *,
                 duty: Optional[float], arm: Optional[str]) -> Dict[str, Any]:
        design = family_key.split("/", 1)[0]
        power = bootstrap_contrast(paired["power"], draws=draws, rng=rng)
        power.update({"quantity": quantity(key, duty=duty, arm=arm,
                                           units=units_of(design)),
                      "class": "MEASURED"})
        if resolution_declared:
            power["sensor_step"] = float(resolution_declared["step_w"])
        if drift_rule is not None:
            trend_shift(power, design, paired["power_offsets"],
                        [1.0] * len(paired["power_offsets"]), "W")
        contrasts[family_key] = power
        return power

    def energy_of(paired: Dict[str, Any], key: str, family_key: str, *,
                  duty: Optional[float], arm: Optional[str]) -> Optional[Dict[str, Any]]:
        if not paired["energy"] or len(paired["energy"]) != len(paired["power"]):
            return None
        design = family_key.split("/", 1)[0]
        row = bootstrap_contrast(paired["energy"], draws=draws, rng=rng)
        row.update({"quantity": quantity(key, duty=duty, arm=arm,
                                         units=units_of(design)),
                    "class": "MEASURED"})
        scales = [float(scale) for _a, _b, scale in paired["energy"]]
        if resolution_declared:
            # One step per inference: the step x the window wall over its inferences.
            row["sensor_step"] = (float(resolution_declared["step_w"])
                                  * statistics.mean(scales))
        if drift_rule is not None:
            trend_shift(row, design, paired["energy_offsets"], scales, "mJ")
        energies.append((row, family_key))
        return row

    for design in sorted({w["design"] for w in protocol}):
        mine = [w for w in protocol if w["design"] == design]
        status = statuses.get(design)
        rebuilt[design] = status == "rebuilt"
        record: Dict[str, Any] = {"xclbin_status": status or "not recorded",
                                  "arms": {}, "controls": {}, "conditions": {}}
        if units_of(design) > 1:
            record["compute_units"] = units_of(design)
        by_round: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for window in mine:
            if window["condition"] != "kernel_idle" and not window.get("passed", False):
                out["excluded_windows"].append({
                    "design": design, "condition": window["condition"],
                    "round": window.get("round"),
                    "reason": "certificates or readouts failed in the window"
                    if window["condition"].startswith("loop_")
                    else "the control's source campaign was not certified"})
                continue
            by_round.setdefault(int(window["round"]), {})[window["condition"]] = window
        conditions = sorted({w["condition"] for w in mine})
        for condition in conditions:
            temps = [temps_in(rows, w["t_start"], w["t_end"]) for w in mine
                     if w["condition"] == condition]
            record["conditions"][condition] = {"windows": len(temps), "temps_c": temps}
        idle_all: List[_Series] = []
        idle_points: List[Tuple[float, float]] = []
        for _number, conds in sorted(by_round.items()):
            if "kernel_idle" in conds:
                idle_s, _ = _series(rows, conds["kernel_idle"], warmup=warmup,
                                    block=block_len)
                if idle_s.n:
                    idle_all.append(idle_s)
                    idle_points.append((_mean_time(rows, conds["kernel_idle"],
                                                   warmup=warmup), idle_s.mean))
        if drift_rule is not None:
            trends[design] = idle_trend(idle_points)
            record["idle_trend"] = trends[design]
        rounds_min = int(gate["rounds_min"])
        # The transfer controls first: a loop's net contrast needs its control.
        for condition in [c for c in conditions if c.startswith("transfer_")]:
            arm = condition[len("transfer_"):]
            paired = _rounds(rows, by_round, condition, "kernel_idle", warmup=warmup,
                             block=block_len, names=("control", "idle"))
            if len(paired["power"]) < rounds_min:
                record["controls"][arm] = {"refused": (
                    f"{len(paired['power'])} usable round(s) < {rounds_min}: no estimate")}
                continue
            family_key = f"{design}/{arm}/transfer_increment_w"
            record["controls"][arm] = {
                "transfer_increment_w": contrast(paired, "transfer_increment_w",
                                                 family_key, duty=None, arm=arm),
                "transfer_energy_per_inference_mJ": energy_of(
                    paired, "transfer_energy_per_inference_mJ", family_key,
                    duty=None, arm=arm),
                "inferences": paired["inferences"], "readings": paired["counts"],
            }
        for condition in [c for c in conditions if c.startswith("loop_")]:
            arm = condition[len("loop_"):]
            paired = _rounds(rows, by_round, condition, "kernel_idle", warmup=warmup,
                             block=block_len, names=("loop", "idle"))
            if len(paired["power"]) < rounds_min:
                record["arms"][arm] = {"refused": (
                    f"{len(paired['power'])} usable round(s) < {rounds_min}: no estimate")}
                continue
            duty = statistics.mean(paired["duty"]) if paired["duty"] else None
            family_key = f"{design}/{arm}/workload_increment_w"
            row: Dict[str, Any] = {
                "workload_increment_w": contrast(paired, "workload_increment_w",
                                                 family_key, duty=duty, arm=arm),
                "workload_energy_per_inference_mJ": energy_of(
                    paired, "workload_energy_per_inference_mJ", family_key,
                    duty=duty, arm=arm),
                "card_energy_per_inference_mJ": (statistics.mean(paired["card"])
                                                 if paired["card"] else None),
                "kernel_duty": duty, "inferences": int(round(paired["inferences"])),
                "readings": paired["counts"],
            }
            if paired["duty_window"]:
                # N units: the busy share of the whole window (sum of run walls over
                # N x the window), beside D, the mean of each unit's own duty.
                row["kernel_duty_window"] = statistics.mean(paired["duty_window"])
            control = f"transfer_{arm}"
            if control in conditions and "transfer_increment_w" in record["controls"].get(arm, {}):
                net = _rounds(rows, by_round, condition, control, warmup=warmup,
                              block=block_len, names=("loop", "control"))
                if len(net["power"]) >= rounds_min:
                    net_key = f"{design}/{arm}/net_of_transfers_w"
                    row["net_of_transfers_w"] = contrast(net, "net_of_transfers_w", net_key,
                                                         duty=duty, arm=arm)
                    row["net_of_transfers_energy_per_inference_mJ"] = energy_of(
                        net, "net_of_transfers_energy_per_inference_mJ", net_key,
                        duty=duty, arm=arm)
                else:
                    row["net_of_transfers_w"] = {"refused": (
                        f"{len(net['power'])} round(s) with both the loop and its "
                        f"control < {rounds_min}: no estimate")}
            record["arms"][arm] = row
        # The user-region increment: every idle window of this design against
        # the shell window, valid only when the reset read back a zero UUID.
        if shells and idle_all:
            shell_s, _ = _series(rows, shells[-1], warmup=warmup, block=block_len)
            if shell_verified is not True:
                record["user_region_increment_w"] = {
                    "quantity": QUANTITIES["user_region_increment_w"],
                    "refused": (
                        "the reset left an xclbin loaded; the shell-only window is "
                        "not shell-only" if shell_verified is False else
                        "no zero UUID was read back after the reset (it was skipped, "
                        "failed or not read), so the shell-only window is not known "
                        "to be shell-only")}
            elif shell_s.n:
                region = bootstrap_contrast([(idle, shell_s, 1.0) for idle in idle_all],
                                            draws=draws, rng=rng)
                region.update({"quantity": QUANTITIES["user_region_increment_w"],
                               "class": "MEASURED", "interleaved": False})
                contrasts[f"{design}/user_region_increment_w"] = region
                record["user_region_increment_w"] = region
        out["designs"][design] = record

    rejected = holm({name: c["p_value"] for name, c in contrasts.items()},
                    float(gate["alpha"]))

    def finish(row: Dict[str, Any], separated: bool, unit: str, design: str) -> None:
        resolution = None
        if resolution_declared and row.get("sensor_step") is not None:
            resolution = {"step": float(row["sensor_step"]),
                          "min_steps": float(resolution_declared.get("number_min_steps", 1))}
        row_drift_ok, reason = drift_ok, None
        if drift_rule is not None:
            judged = row.get("drift_trend")
            if judged is None:
                # Not interleaved: the absolute gate (session-end idle against the pilot).
                reason = ("drift reference outside its band (the session-end idle "
                          "against the pilot, the reference of a contrast that is not "
                          "interleaved)")
            else:
                row_drift_ok = bool(judged["inside_band"])
                reason = (f"the idle trend shifts the contrast by {judged['shift']:.3g} "
                          f"{unit}, more than the {judged['band']:.3g} {unit} the "
                          f"{relative_max:.0%} rule leaves beside the half-width"
                          if judged.get("shift") is not None else
                          f"no idle trend to judge the drift by ({judged.get('why')})")
        outcome, sentence = label(row, separated=separated, relative_max=relative_max,
                                  drift_ok=row_drift_ok, unit=unit, resolution=resolution,
                                  drift_reason=reason)
        if rebuilt.get(design):
            sentence += ("; the xclbin was rebuilt after the archived build, so no "
                         "archived Vivado number describes it")
        row.update({"outcome": outcome, "statement": sentence})

    for name, row in contrasts.items():
        row["holm_separated"] = rejected[name]
        finish(row, rejected[name], unit_of(name), name.split("/", 1)[0])
        units = units_of(name.split("/", 1)[0])
        if units > 1 and not name.endswith("user_region_increment_w"):
            # The per-unit figure: arithmetic on the measured increment of N units.
            row["per_compute_unit"] = {
                "quantity": (f"{row['quantity']}, per compute unit (the increment of "
                             f"{units} units divided by {units})"),
                "point": row["point"] / units, "lo": row["lo"] / units,
                "hi": row["hi"] / units, "half_width": row["half_width"] / units,
                "outcome": row["outcome"], "compute_units": units,
                "class": "MEASURED (a measured increment divided by the declared unit count)",
            }
    for row, family_key in energies:
        # The energy inherits its power contrast's separation (the same rounds).
        finish(row, rejected[family_key], "mJ", family_key.split("/", 1)[0])
    for design, record in out["designs"].items():
        comparison = vivado_comparison(
            record, (designs_declared.get(design) or {}).get("vivado"),
            status=statuses.get(design), units=units_of(design))
        if comparison is not None:
            record["vivado_comparison"] = comparison
    out["family"] = sorted(contrasts)
    return out


def format_estimate(result: Dict[str, Any]) -> str:
    lines = ["design  arm      quantity                                    outcome        statement"]

    def line(design: str, who: str, key: str, row: Optional[Dict[str, Any]]) -> None:
        if not row:
            return
        if "refused" in row:
            lines.append(f"{design:<7} {who:<8} {key:<43} {'REFUSED':<14} {row['refused']}")
        else:
            lines.append(f"{design:<7} {who:<8} {key:<43} {row['outcome']:<14} "
                         f"{row['statement']}")

    for design, record in sorted(result["designs"].items()):
        for arm, row in sorted(record.get("controls", {}).items()):
            if "refused" in row:
                line(design, arm, "transfer_increment_w", row)
                continue
            for key in ("transfer_increment_w", "transfer_energy_per_inference_mJ"):
                line(design, arm, key, row.get(key))
        for arm, row in sorted(record["arms"].items()):
            if "refused" in row:
                line(design, arm, "workload_increment_w", row)
                continue
            for key in ("workload_increment_w", "workload_energy_per_inference_mJ",
                        "net_of_transfers_w", "net_of_transfers_energy_per_inference_mJ"):
                line(design, arm, key, row.get(key))
        line(design, "-", "user_region_increment_w", record.get("user_region_increment_w"))
        comparison = record.get("vivado_comparison")
        if comparison and "refused" in comparison:
            lines.append(f"{design:<7} -        vivado_comparison{'':<26} REFUSED        "
                         f"{comparison['refused']}")
        elif comparison:
            for arm, row in sorted(comparison["arms"].items()):
                lines.append(f"{design:<7} {arm:<8} vivado_comparison{'':<26} "
                             f"{'side by side':<14} {row['reading']}")
        lines.append(f"{design:<7} -        xclbin_status{'':<30} {record['xclbin_status']}")
    warm = result.get("warmup") or {}
    if warm:
        lines.append(f"warm-up: {warm.get('readings')} reading(s) per window "
                     f"({warm.get('seconds', 0.0):.2f} s; {warm.get('rule')})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pilot", "estimate"))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--markers", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pilot", default=None, help="a pilot.json (estimate only)")
    parser.add_argument("--out", default=None)
    options = parser.parse_args(argv)
    with open(options.plan, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if options.command == "pilot":
        result = pilot(options.samples, options.markers, plan)
        print(f"[pilot] verdict {result['verdict']}: {'; '.join(result['reasons'])}")
    else:
        pilot_result = None
        if options.pilot:
            with open(options.pilot, "r", encoding="utf-8") as handle:
                pilot_result = json.load(handle)
        result = estimate(options.samples, options.markers, plan, pilot_result=pilot_result)
        print(format_estimate(result))
    if options.out:
        with open(options.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
