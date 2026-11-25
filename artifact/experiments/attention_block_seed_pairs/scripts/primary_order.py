""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
SEEDS = ("s1", "s2", "s3", "s4", "s5")
PHASES = ("primary", "all")
FAILED = "failed_"

TRACKS = OrderedDict([
    ("headline_v46_pairs", OrderedDict(
        jobs=("F-R18-w100", "F-R18-w071", "F-R18-w050", "F-R18-w035", "F-R10-w100", "F-R10-w050",
              "Q-R18-w100-b4444", "Q-R18-w071-b4444", "Q-R18-w050-b4444", "Q-R18-w035-b4444", "Q-R10-w100-b4444",
              "Q-R10-w050-b4444"),
        primary=("F-R18-w100", "F-R18-w071", "F-R18-w050", "Q-R18-w100-b4444", "Q-R18-w071-b4444",
                 "Q-R18-w050-b4444"),
        depends_on=OrderedDict([("Q-R18-w100-b4444", ("F-R18-w100",)), ("Q-R18-w071-b4444", ("F-R18-w071",)),
                                ("Q-R18-w050-b4444", ("F-R18-w050",)), ("Q-R18-w035-b4444", ("F-R18-w035",)),
                                ("Q-R10-w100-b4444", ("F-R10-w100",)), ("Q-R10-w050-b4444", ("F-R10-w050",))]))),
    ("vit_gate_pairs", OrderedDict(
        jobs=("F", "A", "QF-T", "QF-O8", "QF-O6", "QF-O4", "F-R", "A-R"),
        primary=("F", "A", "QF-T", "QF-O8"),
        depends_on=OrderedDict([("A", ("F",)), ("QF-T", ("F", "A")), ("QF-O8", ("F", "A")), ("QF-O6", ("F", "A")),
                                ("QF-O4", ("F", "A")), ("A-R", ("F-R",))]))),
])


def phase_jobs(track: str, phase: str) -> tuple:
    t = TRACKS[track]
    if phase not in PHASES:
        raise SystemExit(f"REFUSED: phase {phase!r} is not one of {PHASES}")
    return t["primary"] if phase == "primary" else t["jobs"]


def status(track: str, seed: str, job: str, build: Path = BUILD):
    try:
        return json.loads((Path(build) / track / seed / "runs" / job / "result.json").read_text()).get("status")
    except (OSError, ValueError):
        return None


def ended(st) -> bool:
    return st == "ok" or (st is not None and str(st).startswith(FAILED))


def settled_map(track: str, seed: str, statuses: dict) -> dict:
    ""
    deps = TRACKS[track]["depends_on"]
    memo: dict = {}

    def failed_upstream(j: str) -> bool:
        if j in memo:
            return memo[j]
        out = False
        for d in deps.get(j, ()):
            sd = statuses.get(d)
            if (sd is not None and str(sd).startswith(FAILED)) or failed_upstream(d):
                out = True
                break
        memo[j] = out
        return out
    return {j: ended(statuses.get(j)) or failed_upstream(j) for j in TRACKS[track]["jobs"]}


def unsettled(phase: str = "primary", build: Path = BUILD, tracks=None, seeds=SEEDS) -> list:
    ""
    out = []
    for track in (tracks or TRACKS):
        for seed in seeds:
            sts = {j: status(track, seed, j, build) for j in TRACKS[track]["jobs"]}
            sm = settled_map(track, seed, sts)
            out += [f"{track}/{seed}/{j}: {sts[j] or 'no record'}" for j in phase_jobs(track, phase) if not sm[j]]
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    left = unsettled("primary")
    if "--json" in argv:
        print(json.dumps({"primary_unsettled": left, "settled": not left}, indent=1))
    else:
        print("every primary job of both tracks is settled" if not left else
              f"{len(left)} primary jobs not settled:\n  " + "\n  ".join(left[:40]))
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
