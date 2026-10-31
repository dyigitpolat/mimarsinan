#!/usr/bin/env python3
"""The SCK clock-path check: no SCK clock's latency through a register's clock-to-output arc.

Each vendored ODIN core's SPI slave is clocked by SCK, a clock generated on the output of
the ``sck`` register of the kernel's SPI master, which runs on the gated core clock. The
clock that reaches that register is the platform clock through the core-clock gate's
clock arc. The gate also has side inputs, its enable and the platform reset, each driven
by a register. If the generated clock's source latency is traced through one of those
registers' clock-to-output arcs, every check that reads the SCK clock's late latency is
held against a path no SCK edge takes (survey build lock revision 2, defect D-1).

This check reads the "Timing Details" of a routed Vivado timing summary (the whole report,
or any copy of its sections; plain or gzip). For every printed path whose launch clock
("From Clock") or capture clock ("To Clock") is an SCK clock, in every table (Intra, Inter
in both directions, and Other Path Groups such as ``**async_default**``), it reads both
clock paths, the source (launch) and the destination (capture) clock path, and lists every
sequential cell (a flip-flop or latch primitive) whose clock-to-output arc lies on each. The
SCK register's own arc (its output is the SCK clock's own source pin) is the generated
clock's own arc and is not counted. It FAILS on:

  * a clock path of an SCK path that holds any other sequential cell's clock-to-output arc;
  * no printed path of an SCK clock at all (an empty read is never a pass);
  * an SCK clock count other than ``--expect-sck``, when it is given.

It prints every SCK path read (one line each), the count of printed paths that print no
clock path (a ``-datapath_only`` path prints none), and the totals.

    python3 scripts/hacc/check_sck_clock_paths.py <summary.rpt[.gz]>... [--expect-sck N]

An SCK clock is a clock whose name ends in ``sck_reg/Q``: a generated clock takes the name
of the pin it is defined on, and the SPI master's register is ``sck``. Exit 0: PASS. Exit
1: FAIL (each reason on its own line). Exit 2: no "Timing Details" or no path section in an
input. Standard library only: it runs on sura and on the build host alike.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCK_CLOCK = re.compile(r"sck_reg/Q$")
#: Flip-flop and latch primitives (FDRE, FDCE, FDPE, FDSE, LDCE, LDPE and their kin).
SEQUENTIAL_CELL = re.compile(r"^(FD|LD)[A-Z0-9_]*$")
_EDGE = re.compile(r"^\(clock (\S+) (rise|fall) edge\)$")
_ARC = re.compile(r"(\S+) \((Prop_\S+)\)\s*$")
_NET = re.compile(r"\bnet \(fo=\d+(, routed)?\)")
_SLACK = re.compile(r"^Slack(?: \((?:MET|VIOLATED)\))?\s*:\s*(\S+)")
_PROP_TAIL = re.compile(r"_([A-Za-z0-9]+)$")


class NotTimingDetails(ValueError):
    """The input has no "Timing Details" section and no path section."""


@dataclass
class ClockPathElement:
    """One element of a printed clock path, in the order printed."""

    kind: str  # "port", "net", "arc", "pin", "other"
    text: str
    cell: str = ""
    label: str = ""
    routed: bool = False


@dataclass
class ClockPath:
    clock: str
    edge: str
    elements: List[ClockPathElement] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    def pins(self) -> List[str]:
        return [e.text for e in self.elements if e.kind == "pin"]

    def arcs(self) -> List[Tuple[str, str, str, str]]:
        """(cell type, arc label, input pin, output pin) of every cell arc, in order.

        The input pin is the last pin printed before the arc; the output pin is the first
        pin printed after it whose name ends in the arc label's output pin name."""
        out = []
        for i, e in enumerate(self.elements):
            if e.kind != "arc":
                continue
            inp = ""
            for prev in reversed(self.elements[:i]):
                if prev.kind == "pin":
                    inp = prev.text
                    break
                if prev.kind in ("arc", "net"):
                    break
            tail = _PROP_TAIL.search(e.label)
            out_name = tail.group(1) if tail else ""
            outp = ""
            for nxt in self.elements[i + 1:]:
                if nxt.kind == "pin" and out_name and nxt.text.endswith("/" + out_name):
                    outp = nxt.text
                    break
                if nxt.kind in ("arc", "net"):
                    break
            out.append((e.cell, e.label, inp, outp))
        return out

    def sequential_outputs(self) -> List[str]:
        """The output pin (or, if none is printed, the cell type) of every sequential arc."""
        return [outp or f"<{cell}>" for cell, _label, _inp, outp in self.arcs()
                if SEQUENTIAL_CELL.match(cell)]


@dataclass
class TimingPath:
    source: str
    section: int
    group: str
    from_clock: str
    to_clock: str
    path_type: str = "?"
    slack: str = "?"
    exception: str = ""
    start: str = "?"
    end: str = "?"
    clock_paths: List[ClockPath] = field(default_factory=list)

    @property
    def launch(self) -> Optional[ClockPath]:
        return self.clock_paths[0] if len(self.clock_paths) > 0 else None

    @property
    def capture(self) -> Optional[ClockPath]:
        return self.clock_paths[1] if len(self.clock_paths) > 1 else None

    def own_sources(self) -> Tuple[str, ...]:
        """The SCK clocks' own source pins: a generated clock is named after its pin."""
        return tuple(c for c in (self.from_clock, self.to_clock) if SCK_CLOCK.search(c))

    def side_registers(self, which: str) -> List[str]:
        path = self.launch if which == "launch" else self.capture
        if path is None:
            return []
        own = self.own_sources()
        return [p for p in path.sequential_outputs() if p not in own]


def _classify(line: str) -> ClockPathElement:
    text = line.strip()
    arc = _ARC.search(text)
    if arc:
        return ClockPathElement("arc", text, cell=arc.group(1), label=arc.group(2))
    net = _NET.search(text)
    if net:
        return ClockPathElement("net", text.split()[-1], routed=bool(net.group(1)))
    if text.endswith("(IN)"):
        return ClockPathElement("port", text.split()[-2] if len(text.split()) > 1 else text)
    tokens = text.split()
    if tokens and "/" in tokens[-1] and not tokens[-1].startswith("pblock"):
        return ClockPathElement("pin", tokens[-1])
    return ClockPathElement("other", text)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_paths(text: str, source: str = "") -> List[TimingPath]:
    """Every printed path of every section whose launch or capture clock is an SCK clock."""
    lines = [ln[5:] if ln.startswith("#### ") else ln for ln in text.splitlines()]
    if not any(ln.startswith("| Timing Details") or ln.startswith("From Clock:") for ln in lines):
        raise NotTimingDetails(f"{source or 'input'}: no 'Timing Details' and no path section")
    paths: List[TimingPath] = []
    section = 0
    group = ""
    pending_group = ""
    from_clock = to_clock = ""
    in_sck = False
    current: Optional[TimingPath] = None
    phase = ""  # "", "L" (launch clock path) or "C" (capture clock path)
    for line in lines:
        if line.startswith("Path Group:"):
            pending_group = line.split(":", 1)[1].strip()
            continue
        if line.startswith("From Clock:"):
            current = None
            phase = ""
            from_clock = line.split(":", 1)[1].strip()
            group = pending_group or "(clock pair)"
            pending_group = ""
            continue
        if line.startswith("  To Clock:"):
            to_clock = line.split(":", 1)[1].strip()
            section += 1
            in_sck = bool(SCK_CLOCK.search(from_clock) or SCK_CLOCK.search(to_clock))
            continue
        if not in_sck:
            continue
        slack = _SLACK.match(line)
        if slack:
            current = TimingPath(source, section, group, from_clock, to_clock,
                                 slack=slack.group(1))
            paths.append(current)
            phase = ""
            continue
        if current is None:
            continue
        stripped = line.strip()
        if line.startswith("  Source:"):
            current.start = stripped.split()[1] if len(stripped.split()) > 1 else "?"
            continue
        if line.startswith("  Destination:"):
            current.end = stripped.split()[1] if len(stripped.split()) > 1 else "?"
            continue
        if line.startswith("  Path Type:"):
            current.path_type = stripped.split()[2] if len(stripped.split()) > 2 else "?"
            continue
        if line.startswith("  Timing Exception:"):
            current.exception = stripped.split(":", 1)[1].strip()
            continue
        edge = _EDGE.match(stripped)
        if edge:
            current.clock_paths.append(ClockPath(edge.group(1), edge.group(2)))
            phase = "L" if len(current.clock_paths) == 1 else "C"
            continue
        if not phase:
            continue
        if phase == "L" and stripped.startswith("-----"):
            phase = ""
            continue
        if phase == "C" and (stripped.startswith("-----") or "clock pessimism" in stripped
                             or "clock uncertainty" in stripped):
            phase = ""
            continue
        cp = current.clock_paths[-1]
        cp.lines.append(line)
        cp.elements.append(_classify(line))
    return paths


@dataclass
class Verdict:
    paths: List[TimingPath]
    sck_clocks: List[str]
    failures: List[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def check(paths: Sequence[TimingPath], expect_sck: Optional[int] = None) -> Verdict:
    sck_clocks = sorted({c for p in paths for c in p.own_sources()})
    failures: List[str] = []
    if not paths:
        failures.append("no path of an SCK clock is printed: an empty read is not a pass")
    if expect_sck is not None and len(sck_clocks) != expect_sck:
        failures.append(f"{len(sck_clocks)} SCK clock(s) in the printed paths, "
                        f"expected {expect_sck}")
    bad: Dict[Tuple[str, str], int] = {}
    for p in paths:
        for which in ("launch", "capture"):
            for reg in p.side_registers(which):
                bad[(which, reg)] = bad.get((which, reg), 0) + 1
    for (which, reg), n in sorted(bad.items()):
        failures.append(f"{n} {which} clock path(s) of SCK paths carry the clock-to-output "
                        f"arc of {reg}")
    return Verdict(list(paths), sck_clocks, failures)


def _cells(regs: List[str]) -> str:
    return ", ".join(regs) if regs else "none"


def render(verdict: Verdict) -> str:
    out = []
    for p in verdict.paths:
        n_clock = len(p.clock_paths)
        out.append(
            f"{p.source} | sec {p.section} | group {p.group} | from {p.from_clock} | to "
            f"{p.to_clock} | {p.path_type} | slack {p.slack} | exception "
            f"{p.exception or 'none'} | {p.start} -> {p.end} | clock paths {n_clock} | "
            f"launch side: {_cells(p.side_registers('launch'))} | "
            f"capture side: {_cells(p.side_registers('capture'))}")
    total = len(verdict.paths)
    none = sum(1 for p in verdict.paths if not p.clock_paths)
    both = sum(1 for p in verdict.paths if len(p.clock_paths) >= 2)
    launch = sum(1 for p in verdict.paths if p.side_registers("launch"))
    capture = sum(1 for p in verdict.paths if p.side_registers("capture"))
    out.append(f"SCK clocks: {len(verdict.sck_clocks)}")
    out.append(f"paths printed: {total}; with no clock path: {none}; with both clock paths: "
               f"{both}")
    out.append(f"clock paths with a side register: launch {launch}, capture {capture}, "
               f"total {launch + capture}")
    for reason in verdict.failures:
        out.append(f"FAIL: {reason}")
    out.append("RESULT: " + ("PASS" if verdict.passed else "FAIL"))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("summary", nargs="+", type=Path,
                    help="routed timing summary or copies of its sections (plain or gzip)")
    ap.add_argument("--expect-sck", type=int, default=None,
                    help="the number of SCK clocks the build has (units x cores)")
    args = ap.parse_args(argv)
    paths: List[TimingPath] = []
    try:
        for source in args.summary:
            paths.extend(parse_paths(read_text(source), str(source)))
    except NotTimingDetails as exc:
        print(f"NOT A TIMING SUMMARY: {exc}")
        return 2
    verdict = check(paths, args.expect_sck)
    print(render(verdict))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
