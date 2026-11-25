#!/usr/bin/env python3
"""The SPI clock-domain gate: read a routed ODIN build's timing summary for the SCK domains.

Each vendored ODIN core is programmed through an SPI slave clocked by SCK, the output
of the kernel SPI master's ``sck`` register. A build that declares no clock on it
leaves the domain untimed, and the routed summary says so in its own ``check_timing``:
register pins with no clock, driven by ``.../spi_i/sck_reg/Q``, and unconstrained
endpoints. This gate reads the summary Vivado writes after routing
(``prj.runs/<run>/hw_bb_locked_timing_summary_routed.rpt``, or its part before
"Timing Details") and FAILS on any of:

  * a register pin with no clock (each root named; an SPI root is ``*spi_i/sck_reg/Q``);
  * an unconstrained internal endpoint. The summary gives only the count, and a
    build without the SPI slave (the widened fabric) reads 0, so every one fails;
  * an SCK clock count other than ``--expect-sck`` (units x cores). The count is
    required whenever the summary shows an SPI domain (an SPI root or an SCK clock),
    so a build that declares only some of its SCK clocks cannot pass without it; a
    fabric with no SPI slave (a generated core) shows neither and needs none;
  * an SCK clock with no timed path inside its domain, or with none from or to the
    core clock it is generated from (a false path or clock group would hide them);
  * an SCK clock whose own row (Intra Clock Table) has no number for WNS or WHS: a
    setup or hold exception inside the domain blanks that cell;
  * a crossing row into or out of an SCK clock (Inter Clock Table) with no number for
    WNS or WHS: an exception on the crossing blanks that cell (``set_false_path -hold``
    from the core clock into SCK would hide MOSI's hold check);
  * a negative setup or hold slack inside an SCK domain, across any of its crossings,
    or on its asynchronous (recovery and removal) paths; and a negative design WNS, WHS
    or WPWS.

It prints the setup and hold slack of every SCK clock and its master for the record.

    python3 scripts/hacc/check_spi_timing.py <summary.rpt> [--expect-sck N] [--json out.json]

Exit 0: PASS. Exit 1: FAIL (each reason on its own line). Exit 2: not a routed timing
summary. Standard library only: it runs on a workstation and on the build host alike.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

SCK_ROOT = re.compile(r"spi_i/sck_reg/Q$")
SCK_CLOCK = re.compile(r"spi_i/sck_reg")
_CHECK = re.compile(r"^\s*\d+\.\s+checking\s+(\w+)\s+\((\d+)\)")
_NO_CLOCK = re.compile(
    r"There are (\d+) register/latch pins with no clock driven by root clock pin: (\S+)")
_WAVE = re.compile(r"^(\s*)(\S+)\s+\{[^}]*\}\s+(\S+)\s+(\S+)\s*$")

Row = Dict[str, Optional[str]]


class NotASummary(ValueError):
    """The file is not a Vivado timing summary with a check_timing section."""


def _section(lines: Sequence[str], title: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.rstrip() == f"| {title}":
            return i
    return None


def _table(lines: Sequence[str], title: str, names: int) -> List[Row]:
    """Rows of one summary table, cells keyed by header; blank cells are None.

    Name columns are left-aligned at their dash run's start; numbers are
    right-aligned at their dash run's end, so a cell is the token ending there.
    """
    start = _section(lines, title)
    if start is None:
        return []
    i = start + 1
    while i < len(lines) and not re.match(r"^\s*-+(\s+-+)+\s*$", lines[i]):
        i += 1
    if i >= len(lines):
        return []
    header, dashes = lines[i - 1], lines[i]
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", dashes)]
    keys = [header[a:b].strip() if k >= names else header[a:].split("  ")[0].strip()
            for k, (a, b) in enumerate(spans)]
    rows: List[Row] = []
    for line in lines[i + 1:]:
        if not line.strip():
            break
        tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", line)]
        row: Row = {}
        for k, (a, b) in enumerate(spans):
            if k < names:
                limit = spans[k + 1][0] if k + 1 < len(spans) else len(line)
                found = [t for t in tokens if a <= t[0] < limit]
                row[keys[k]] = found[0][2] if found else None
            else:
                found = [t for t in tokens if t[1] == b]
                row[keys[k]] = found[0][2] if found else None
        rows.append(row)
    return rows


def _num(value: Optional[str]) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except ValueError:
        return None


def parse(text: str) -> Dict[str, object]:
    lines = text.splitlines()
    checks: Dict[str, int] = {}
    roots: List[Tuple[str, int]] = []
    for line in lines:
        m = _CHECK.match(line)
        if m and m.group(1) not in checks:
            checks[m.group(1)] = int(m.group(2))
        m = _NO_CLOCK.search(line)
        if m and (m.group(2), int(m.group(1))) not in roots:
            roots.append((m.group(2), int(m.group(1))))
    if "no_clock" not in checks or "unconstrained_internal_endpoints" not in checks:
        raise NotASummary("no check_timing section (no_clock, unconstrained_internal_endpoints)")
    clocks: Dict[str, Optional[str]] = {}
    stack: List[Tuple[int, str]] = []
    start = _section(lines, "Clock Summary")
    for line in lines[start + 1:] if start is not None else []:
        if re.match(r"^\| [A-Za-z]", line):  # the next section's title
            break
        m = _WAVE.match(line)
        if not m:
            continue
        depth, name = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        clocks[name] = stack[-1][1] if stack else None
        stack.append((depth, name))
    design = _table(lines, "Design Timing Summary", 0)
    return {
        "checks": checks, "no_clock_roots": roots, "clocks": clocks,
        "design": design[0] if design else {},
        "intra": _table(lines, "Intra Clock Table", 1),
        "inter": _table(lines, "Inter Clock Table", 2),
        "other": _table(lines, "Other Path Groups Table", 3),
    }


def _negative(row: Row, *keys: str) -> List[str]:
    return [f"{key} {row.get(key)}" for key in keys
            if (_num(row.get(key)) is not None and (_num(row.get(key)) or 0.0) < 0.0)]


def _untimed(row: Row, where: str) -> List[str]:
    """WNS and WHS cells with no number: an exception (a false path, a clock group) removed it."""
    return [f"{key} has no number: the {check} check {where} is not timed"
            for key, check in (("WNS(ns)", "setup"), ("WHS(ns)", "hold"))
            if _num(row.get(key)) is None]


def verdict(summary: Dict[str, object], expect_sck: Optional[int] = None) -> List[str]:
    """Every reason the summary fails the gate; an empty list is a PASS."""
    fails: List[str] = []
    checks = summary["checks"]
    assert isinstance(checks, dict)
    roots = summary["no_clock_roots"]
    assert isinstance(roots, list)
    if checks["no_clock"]:
        fails.append(f"{checks['no_clock']} register pin(s) with no clock")
        for root, count in roots:
            kind = "SPI SCK" if SCK_ROOT.search(root) else "other"
            fails.append(f"  {count} unclocked pin(s) driven by {root} ({kind})")
    if checks["unconstrained_internal_endpoints"]:
        fails.append(f"{checks['unconstrained_internal_endpoints']} unconstrained internal "
                     f"endpoint(s)")
    clocks = summary["clocks"]
    assert isinstance(clocks, dict)
    scks = [name for name in clocks if SCK_CLOCK.search(name)]
    spi_roots = [root for root, _ in roots if SCK_ROOT.search(root)]
    if expect_sck is None:
        if scks or spi_roots:
            fails.append(f"SPI clock domain in the summary ({len(scks)} SCK clock(s), "
                         f"{len(spi_roots)} unclocked SPI root(s)) but no expected SCK count "
                         f"(--expect-sck)")
    elif len(scks) != expect_sck:
        fails.append(f"{len(scks)} SCK clock(s) in the clock summary, expected {expect_sck}")
    intra = {row["Clock"]: row for row in summary["intra"]}  # type: ignore[union-attr]
    for sck in scks:
        master = clocks[sck]
        row = intra.get(sck)
        if row is None:
            fails.append(f"{sck}: no timed path inside its domain")
        else:
            fails += [f"{sck}: intra {item}" for item in _untimed(row, "inside the domain")]
            fails += [f"{sck}: intra {item}" for item in _negative(row, "WNS(ns)", "WHS(ns)")]
        for a, b in ((master, sck), (sck, master)):
            pair = [r for r in summary["inter"] if (r["From Clock"], r["To Clock"]) == (a, b)]  # type: ignore[union-attr]
            if not pair:
                fails.append(f"{a} -> {b}: no timed path across the SPI crossing")
            for r in pair:
                fails += [f"{a} -> {b}: {item}" for item in _untimed(r, "across the SPI crossing")]
                fails += [f"{a} -> {b}: {item}" for item in _negative(r, "WNS(ns)", "WHS(ns)")]
    masters = {(clocks[s], s) for s in scks} | {(s, clocks[s]) for s in scks}
    for r in summary["inter"]:  # type: ignore[union-attr]
        a, b = r["From Clock"], r["To Clock"]
        if (a, b) in masters or not any(SCK_CLOCK.search(c or "") for c in (a, b)):
            continue  # a crossing with its master is checked above; this is any other one
        fails += [f"{a} -> {b}: {item}" for item in _untimed(r, "across the SPI crossing")]
        fails += [f"{a} -> {b}: {item}" for item in _negative(r, "WNS(ns)", "WHS(ns)")]
    for r in summary["other"]:  # type: ignore[union-attr]
        if any(SCK_CLOCK.search(r.get(k) or "") for k in ("From Clock", "To Clock")):
            fails += [f"{r['Path Group']} {r['From Clock']} -> {r['To Clock']}: {item}"
                      for item in _negative(r, "WNS(ns)", "WHS(ns)")]
    design = summary["design"]
    assert isinstance(design, dict)
    fails += [f"design {item}" for item in _negative(design, "WNS(ns)", "WHS(ns)", "WPWS(ns)")]
    return fails


def record(summary: Dict[str, object]) -> List[str]:
    """The per-clock lines the build record keeps: every SCK clock and its master."""
    clocks = summary["clocks"]
    assert isinstance(clocks, dict)
    intra = {row["Clock"]: row for row in summary["intra"]}  # type: ignore[union-attr]
    names = [n for n in clocks if SCK_CLOCK.search(n)]
    names = sorted({clocks[n] for n in names if clocks[n]}) + names
    lines = []
    for name in names:
        row = intra.get(name, {})
        lines.append(f"clock {name}: WNS {row.get('WNS(ns)')} WHS {row.get('WHS(ns)')} "
                     f"(endpoints {row.get('TNS Total Endpoints')})")
    for r in summary["inter"]:  # type: ignore[union-attr]
        if any(SCK_CLOCK.search(r.get(k) or "") for k in ("From Clock", "To Clock")):
            lines.append(f"inter {r['From Clock']} -> {r['To Clock']}: WNS {r['WNS(ns)']} "
                         f"WHS {r['WHS(ns)']}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="routed timing summary (.rpt)")
    parser.add_argument("--expect-sck", type=int, default=None,
                        help="SCK clocks the build must carry (compute units x cores); "
                             "required when the summary shows an SPI domain")
    parser.add_argument("--json", default=None, help="write the parsed record here")
    options = parser.parse_args(argv)
    raw = open(options.summary, "rb").read()
    print(f"[spi-timing] report {options.summary} sha256 {hashlib.sha256(raw).hexdigest()}")
    try:
        summary = parse(raw.decode("utf-8", errors="replace"))
    except NotASummary as error:
        print(f"[spi-timing] REFUSING: {error}", file=sys.stderr)
        return 2
    checks = summary["checks"]
    assert isinstance(checks, dict)
    print(f"[spi-timing] check_timing: no_clock {checks['no_clock']}, "
          f"unconstrained_internal_endpoints {checks['unconstrained_internal_endpoints']}")
    for line in record(summary):
        print(f"[spi-timing] {line}")
    fails = verdict(summary, options.expect_sck)
    for line in fails:
        print(f"FAIL: {line}")
    if options.json:
        with open(options.json, "w") as handle:
            json.dump({"summary": summary, "fails": fails}, handle, indent=1, sort_keys=True)
    print(f"RESULT: {'FAIL' if fails else 'PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
