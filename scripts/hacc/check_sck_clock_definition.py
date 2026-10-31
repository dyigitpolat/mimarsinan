#!/usr/bin/env python3
"""The SCK clock-definition check: each SCK clock is traced along the clock, positively.

Each vendored ODIN core's SPI slave is clocked by SCK, a clock generated on the output of
the ``sck`` register of the kernel's SPI master, on the gated core clock. The required
form (survey build lock revision 2, defect D-1, unit U-SP4) defines that clock at a pin
only the clock arc of the core-clock gate reaches, so its latency follows the clock and no
side input of the gate (the enable register, the platform reset register) enters it. The
clock-path check (``check_sck_clock_paths.py``) shows that no register's arc is printed on
an SCK clock path; a form that shortens the latency by declaration or leaves it untraced
would print none either. This check is the positive half. It reads the clock records of a
read-only probe of a routed checkpoint (one folder, the probe's file names):

  ``resolved.txt``            what each generated clock's ``-source`` in the text resolved to
  ``sck_clock_properties.txt`` ``report_property`` of every SCK clock (``== <clock>`` blocks)
  ``master_clocks.txt``        the clock at each SCK register's clock pin and its source pins
  ``source_fanin.txt``         the timing fan-in of each SCK clock's SOURCE pin
  ``feedback_paths.rpt``       the SCK register's feedback, setup and hold, full clock paths
  ``timing_summary.rpt``       (or ``.rpt.gz``, as the records keep it, or ``.head.rpt``) for
                               its ``check_timing``
  ``probe_mode.txt``           (``--own-build`` only) the probe's mode
  ``vivado.log``               (or ``.log.gz``; ``--own-build`` only) the probe's Vivado log

and FAILS, per SCK clock, unless:

  1. IS_GENERATED 1, IS_PROPAGATED 1, DIVIDE_BY 4, MASTER_CLOCK the kernel clock (the clock
     at the SCK register's clock pin), and SOURCE the one pin the text's ``-source``
     resolved to, which is on no clock buffer, and whose timing fan-in holds no register
     (a flip-flop or latch, PRIMITIVE_GROUP REGISTER) other than the pin's own cell and
     reaches the master clock's source;
  2. ``check_timing`` reads ``generated_clocks (0)``;
  3. the feedback's launch clock path, for setup and for hold, is printed with a slack and
     runs from the master's input port, through SOURCE and its own cell's arc, then a clock
     buffer, to the SCK register's clock pin: no register's clock-to-output arc, every net
     after the first cell arc routed, and no latency or ideal-clock line.

With ``--own-build`` the records must be a build's own clocks, so that what the build
resolved at link is checked, not a probe's text: a probe that reads a text replaces every
SCK clock the checkpoint restored (``[Constraints 18-619] ... overwriting the previous
clock``), and its records then show how the text resolves in the routed netlist. It FAILS
also unless:

  4. ``probe_mode.txt`` reads ``mode own`` (the probe read no constraint text) and names the
     build's packaged ``odin_spi_domain.xdc`` as its text (the one whose resolution in the
     routed netlist condition 1 compares SOURCE with); the probe's ``vivado.log`` shows the
     constraints restored from the checkpoint's binary archive and no clock replaced after
     it; and no SCK clock carries a FILE_NAME. Vivado 2022.2.2 restores a routed
     checkpoint's clocks without FILE_NAME or LINE_NUMBER, and a clock defined by a text
     read in the session carries that text's name (MEASURED, U-SP4 targeted pass: probe
     71259 in mode own against the control 71238).

It prints, per SCK clock, its defining file and line, the source, and for each feedback
check its slack and source clock delay (SCD), and the path's shape.

    python3 scripts/hacc/check_sck_clock_definition.py <probe dir> [--expect-sck N] [--own-build]

Exit 0: PASS. Exit 1: FAIL (each reason on its own line). Exit 2: a record is missing.
Standard library only: it runs on sura and on the build host alike.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sck_clock_paths as clock_paths  # noqa: E402  (the same folder; standard library only)

RECORDS = ("resolved.txt", "sck_clock_properties.txt", "master_clocks.txt",
           "source_fanin.txt", "feedback_paths.rpt")
SUMMARIES = ("timing_summary.rpt", "timing_summary.rpt.gz", "timing_summary.head.rpt")
LOGS = ("vivado.log", "vivado.log.gz")
BUILD_XDC = "odin_spi_domain.xdc"  # the SPI clock-domain file build_xclbn.sh packages
_RESTORED = "Restoring constraints from binary archive"
_REPLACED = re.compile(r"\[Constraints 18-619\] A clock with name '([^']+)' already exists")
_KEYS = re.compile(r"\b(pin|cell|ref|group|ref_pin|net|type|driver|driver_ref|driver_group|seq)"
                   r" (\S*)")
_GENERATED = re.compile(r"^\s*\d+\.\s+checking\s+generated_clocks\s+\((\d+)\)", re.M)
_SCD = re.compile(r"Source Clock Delay\s+\(SCD\):\s+(-?[\d.]+ns)")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?ns$")
_LATENCY = re.compile(r"latency|ideal", re.I)
_BUFFER = re.compile(r"^BUF")
_CELL_ARC = re.compile(r"(\S+) \((Prop_\S+)\)\s*$")
EXPECTED = {"IS_GENERATED": "1", "IS_PROPAGATED": "1", "DIVIDE_BY": "4"}


class NotAProbeRecord(ValueError):
    """A record file of the probe is missing."""


def _fields(text: str) -> Dict[str, str]:
    return {k: v for k, v in _KEYS.findall(text)}


@dataclass
class Master:
    """The clock at an SCK register's clock pin (the kernel clock) and that clock's source pins."""

    clock: str
    master: str
    sources: List[str]


@dataclass
class FanIn:
    """The timing fan-in of an SCK clock's SOURCE pin: cells (name, fields) and startpoints."""

    source: str
    cells: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)
    startpoints: List[str] = field(default_factory=list)


@dataclass
class Records:
    resolved: Dict[str, Tuple[int, List[Dict[str, str]]]] = field(default_factory=dict)
    properties: Dict[str, Dict[str, str]] = field(default_factory=dict)
    no_clock: List[str] = field(default_factory=list)
    masters: Dict[str, Master] = field(default_factory=dict)
    fanin: Dict[str, FanIn] = field(default_factory=dict)
    feedback: Dict[Tuple[str, str], str] = field(default_factory=dict)
    summary: str = ""
    mode: Optional[str] = None
    mode_text: Optional[str] = None
    vivado_log: Optional[str] = None


def read_records(folder: Path) -> Records:
    missing = [name for name in RECORDS if not (folder / name).is_file()]
    summary = next((folder / s for s in SUMMARIES if (folder / s).is_file()), None)
    if missing or summary is None:
        raise NotAProbeRecord(f"{folder}: missing {', '.join(missing or ['timing_summary'])}")
    rec = Records(summary=clock_paths.read_text(summary))
    if (folder / "probe_mode.txt").is_file():
        lines = (folder / "probe_mode.txt").read_text().splitlines()
        rec.mode = (lines or [""])[0].strip()
        rec.mode_text = next((ln.split()[1] for ln in lines[1:]
                              if ln.startswith("text ") and len(ln.split()) > 1), None)
    log = next((folder / n for n in LOGS if (folder / n).is_file()), None)
    if log is not None:
        rec.vivado_log = clock_paths.read_text(log)
    target = ""
    for line in (folder / "resolved.txt").read_text().splitlines():
        m = re.match(r"^resolved \S+ line \d+ target (\S+) source_count (\d+)", line)
        if m:
            target = m.group(1)
            rec.resolved[target] = (int(m.group(2)), [])
        elif line.strip().startswith("source ") and target:
            rec.resolved[target][1].append(_fields(line.strip()[len("source "):]))
    clock = ""
    for line in (folder / "sck_clock_properties.txt").read_text().splitlines():
        if line.startswith("== (no clock on "):
            rec.no_clock.append(line[len("== (no clock on "):].rstrip(")"))
            clock = ""
        elif line.startswith("== "):
            clock = line[3:].strip()
            rec.properties[clock] = {}
        elif clock and line.strip() and not line.startswith("Property "):
            parts = line.split(None, 3)
            if len(parts) >= 3:
                rec.properties[clock][parts[0]] = parts[3].strip() if len(parts) == 4 else ""
    for line in (folder / "master_clocks.txt").read_text().splitlines():
        m = re.match(r"^sck_pin (\S+) clock (\S*) master_at_C (\S*) master_source_pins ?(.*)$",
                     line)
        if m:
            rec.masters[m.group(1)] = Master(m.group(2), m.group(3), m.group(4).split())
    clock = ""
    for line in (folder / "source_fanin.txt").read_text().splitlines():
        m = re.match(r"^== (\S+) SOURCE ?(\S*)", line)
        if m:
            clock = m.group(1)
            rec.fanin[clock] = FanIn(m.group(2))
        elif clock and line.startswith("cell "):
            name = line.split()[1]
            rec.fanin[clock].cells.append((name, _fields(line[len("cell ") + len(name):])))
        elif clock and line.startswith("startpoint "):
            rec.fanin[clock].startpoints.append(line.split()[1])
    text = (folder / "feedback_paths.rpt").read_text(errors="replace")
    for chunk in re.split(r"(?m)^(?=#### feedback )", text):
        m = re.match(r"^#### feedback (setup|hold) (\S+)", chunk)
        if m:
            rec.feedback[(m.group(2), m.group(1))] = chunk
    return rec


def _launch_findings(clock: str, check: str, chunk: Optional[str], source: str,
                     master_sources: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(printed lines, failures) for one feedback check's launch clock path."""
    where = f"{clock}: feedback {check}"
    if chunk is None:
        return [], [f"{where}: no path printed"]
    paths = clock_paths.parse_paths(f"From Clock:  {clock}\n  To Clock:  (feedback)\n" + chunk,
                                    "feedback_paths.rpt")
    if len(paths) != 1:
        return [], [f"{where}: no path printed" if not paths else f"{where}: {len(paths)} paths"]
    path = paths[0]
    scd = _SCD.search(chunk)
    printed = [f"{where}: slack {path.slack} SCD {scd.group(1) if scd else '?'}"]
    fails = []
    if not _NUMBER.match(path.slack):
        fails.append(f"{where}: no number for the slack")
    lp = path.launch
    if lp is None:
        return printed, fails + [f"{where}: no launch clock path printed"]
    if lp.clock != clock:
        fails.append(f"{where}: the launch clock is {lp.clock}, not {clock}")
    ports = [e.text for e in lp.elements if e.kind == "port"]
    if not ports or ports[0] not in master_sources:
        fails.append(f"{where}: the launch clock path does not start at the master's input "
                     f"{' '.join(master_sources) or '?'}")
    for line in lp.lines:
        if _LATENCY.search(line):
            fails.append(f"{where}: the launch clock path prints a latency or ideal clock line: "
                         f"{' '.join(line.split())}")
    own = clock_paths.SCK_CLOCK.search(clock) and clock
    for reg in lp.sequential_outputs():
        if reg != own:
            fails.append(f"{where}: the launch clock path carries the clock-to-output arc of {reg}")
    seen_arc = False
    for e in lp.elements:
        if e.kind == "arc":
            seen_arc = True
        elif e.kind == "net" and seen_arc and not e.routed:
            fails.append(f"{where}: the launch clock path has an unrouted net {e.text}")
    arcs = lp.arcs()
    gate = next((i for i, arc in enumerate(arcs) if _leaves(arc, source)), None)
    end = clock[:-2] + "/C" if clock.endswith("/Q") else "?"
    ends = _ends_at(lp, chunk, clock, end)
    if gate is None:
        fails.append(f"{where}: the launch clock path does not leave SOURCE {source} by its own "
                     "cell's arc")
    else:
        buffers = [cell for cell, _l, _i, _o in arcs[gate + 1:] if _BUFFER.match(cell)]
        if not buffers:
            fails.append(f"{where}: no clock buffer after SOURCE {source}")
        printed.append(f"{where}: through {source} by {arcs[gate][0]} ({arcs[gate][1]}), then "
                       f"{', '.join(buffers) or 'no buffer'}, to {end if ends else '?'}")
    if not ends:
        fails.append(f"{where}: the launch clock path does not end at {end}")
    return printed, fails


def _leaves(arc: Tuple[str, str, str, str], source: str) -> bool:
    """The arc is SOURCE's own cell's arc out of SOURCE, a combinational one.

    A report prints a cell's input pin only for some arcs, so the arc is known by its cell
    (its output pin's cell is SOURCE's cell) and by its label, which names the input pin
    (``Prop_<site>_<input>_<output>``), or by the input pin when it is printed."""
    cell, label, inp, outp = arc
    if clock_paths.SEQUENTIAL_CELL.match(cell):
        return False
    if inp == source:
        return True
    src_cell, _, src_pin = source.rpartition("/")
    out_cell, _, out_pin = outp.rpartition("/")
    return bool(src_cell) and out_cell == src_cell and label.endswith(f"_{src_pin}_{out_pin}")


def _ends_at(lp: "clock_paths.ClockPath", chunk: str, clock: str, end: str) -> bool:
    """The launch clock path ends at the SCK register's clock pin ``end``: printed as its last
    pin, or, when the report prints no input pin there, shown by the data path opening with
    that register's own clock-to-output arc onto the SCK clock's source pin."""
    pins = lp.pins()
    if pins and pins[-1] == end:
        return True
    lines = chunk.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith(f"(clock {clock} ") and ln.strip().endswith("edge)")),
                 None)
    if start is None:
        return False
    sep = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("-----")),
               None)
    if sep is None:
        return False
    end_pin = end.rpartition("/")[2]
    for i in range(sep + 1, len(lines)):
        arc = _CELL_ARC.search(lines[i].strip())
        if not arc:
            continue
        q_name = clock.rpartition("/")[2]
        if not arc.group(2).endswith(f"_{end_pin}_{q_name}"):
            return False
        for nxt in lines[i + 1:]:
            tokens = nxt.split()
            if tokens and "/" in tokens[-1] and not tokens[-1].startswith("pblock"):
                return tokens[-1] == clock
        return False
    return False


def _own_build_findings(rec: Records) -> List[str]:
    """Condition 4's record-wide part: the records are the build's own restored clocks."""
    fails = []
    if rec.mode != "mode own":
        state = "is absent" if rec.mode is None else f"reads '{rec.mode}'"
        fails.append(f"the records are not a read of the build's own clocks: probe_mode.txt {state}, "
                     "not 'mode own'")
    elif rec.mode_text is None or Path(rec.mode_text).name != BUILD_XDC:
        fails.append(f"the own read's text is {rec.mode_text or 'not named'}, not the build's "
                     f"packaged {BUILD_XDC}")
    if rec.vivado_log is None:
        fails.append("the probe's vivado.log is absent: the restore of the build's constraints is "
                     "not shown")
    else:
        if _RESTORED not in rec.vivado_log:
            fails.append(f"the probe's vivado.log shows no '{_RESTORED}'")
        for name in _REPLACED.findall(rec.vivado_log):
            fails.append(f"the probe's vivado.log shows clock {name} replaced after the restore "
                         "([Constraints 18-619])")
    return fails


def check(rec: Records, expect_sck: Optional[int] = None,
          own_build: bool = False) -> Tuple[List[str], List[str]]:
    printed: List[str] = []
    fails: List[str] = [f"no clock on {pin}" for pin in rec.no_clock]
    if own_build:
        fails.extend(_own_build_findings(rec))
    if expect_sck is not None and len(rec.properties) != expect_sck:
        fails.append(f"{len(rec.properties)} SCK clock(s) in the records, expected {expect_sck}")
    for clock, props in rec.properties.items():
        q_pin = props.get("SOURCE_PINS", clock).split()[0] if props.get("SOURCE_PINS") else clock
        file_name = props.get("FILE_NAME", "")
        printed.append(f"{clock}: defined in {file_name or '(no FILE_NAME)'} line "
                       f"{props.get('LINE_NUMBER') or '(no LINE_NUMBER)'}")
        if own_build and file_name:
            fails.append(f"{clock}: FILE_NAME is {file_name}: the clock was defined by a text the "
                         "probe read, not restored from the build's checkpoint")
        for prop, want in EXPECTED.items():
            if props.get(prop) != want:
                fails.append(f"{clock}: {prop} is {props.get(prop, 'absent')}, not {want}")
        master = rec.masters.get(q_pin, Master("", "", []))
        kernel = master.master
        if not kernel or props.get("MASTER_CLOCK") != kernel:
            fails.append(f"{clock}: MASTER_CLOCK is {props.get('MASTER_CLOCK', 'absent')}, not the "
                         f"kernel clock {kernel or '?'}")
        count, sources = rec.resolved.get(q_pin, (0, []))
        if count != 1 or len(sources) != 1:
            fails.append(f"{clock}: the text's source resolves to {count} pins, not 1")
        declared = sources[0] if sources else {}
        source = props.get("SOURCE", "")
        if declared and source != declared.get("pin"):
            fails.append(f"{clock}: SOURCE is {source or 'absent'}, not the declared "
                         f"{declared.get('pin')}")
        if declared:
            printed.append(f"{clock}: SOURCE {source} ({declared.get('ref')}, group "
                           f"{declared.get('group')}, net type {declared.get('type')})")
            if declared.get("group") == "CLOCK":
                fails.append(f"{clock}: the declared source {declared.get('pin')} is on a clock "
                             f"buffer ({declared.get('ref')}), not the gate")
        fan = rec.fanin.get(clock)
        if fan is None:
            fails.append(f"{clock}: no fan-in record of SOURCE {source}")
        else:
            own_cell = declared.get("cell", "")
            regs = [name for name, f in fan.cells
                    if f.get("group") == "REGISTER" and name != own_cell]
            if regs:
                fails.append(f"{clock}: the fan-in of SOURCE {source} holds register(s) "
                             f"{', '.join(regs)}")
            if not set(master.sources) & set(fan.startpoints):
                fails.append(f"{clock}: the fan-in of SOURCE {source} does not reach the master's "
                             f"source {' '.join(master.sources) or '?'}")
        for check_name in ("setup", "hold"):
            p, f = _launch_findings(clock, check_name, rec.feedback.get((q_pin, check_name)),
                                    source, master.sources)
            printed.extend(p)
            fails.extend(f)
    generated = _GENERATED.search(rec.summary)
    if generated is None:
        fails.append("check_timing has no generated_clocks line")
    else:
        printed.append(f"check_timing generated_clocks ({generated.group(1)})")
        if generated.group(1) != "0":
            fails.append(f"check_timing reads generated_clocks ({generated.group(1)}), not (0)")
    return printed, fails


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("records", type=Path, help="the probe's output folder")
    ap.add_argument("--expect-sck", type=int, default=None,
                    help="the number of SCK clocks the build has (units x cores)")
    ap.add_argument("--own-build", action="store_true",
                    help="the records must be the build's own clocks as its checkpoint restored "
                         f"them (probe mode own, no clock read), compared with its {BUILD_XDC}")
    args = ap.parse_args(argv)
    try:
        rec = read_records(args.records)
    except NotAProbeRecord as exc:
        print(f"NOT A PROBE RECORD: {exc}")
        return 2
    printed, fails = check(rec, args.expect_sck, args.own_build)
    print(f"L3 SCK clock-definition check: {args.records}")
    what = "the build's own clocks (--own-build)" if args.own_build else "the probe's records"
    print(f"read: {what}; probe {rec.mode or 'mode not recorded'}")
    print(f"SCK clocks: {len(rec.properties)}")
    for line in printed:
        print(line)
    for reason in fails:
        print(f"FAIL: {reason}")
    print("RESULT: " + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
