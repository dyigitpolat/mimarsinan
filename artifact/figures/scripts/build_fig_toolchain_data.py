""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
F2_SCRIPTS = F3.parent / "figures_v2" / "scripts"
sys.path.insert(0, str(F2_SCRIPTS))
import _ledger as LG
from build_fig_toolchain_1b_data import decision_classes

NAME = "fig_toolchain"
SECTIONS = LG.D5 / "sections"
TOOLCHAIN = SECTIONS / "toolchain.tex"
THEORY = SECTIONS / "theory.tex"


D6_VERDICTS = LG.D5.parent / "v6_visual" / "sections" / "front.tex"


V6_OUTCOMES_SENTENCE = (r"Passing both makes it \\emph\{bit-exact\}.*?Passing only the sampled run makes it "
                        r"\\emph\{built and checked\}\. A differing output yields a named refusal")
V6_OUTCOME_NAME = {
    "Certified": "Bit-exact",
    "Built and checked": "Built and checked",
    "Refused by name": "Named refusal",
}


def _flat(path: Path):
    ""
    text, starts = "", []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        ln = re.sub(r"(?<!\\)%.*$", "", ln).strip()
        starts.append((len(text), i))
        text += ln + " "
    return text, starts


def quote(path: Path, pattern: str) -> dict:
    ""
    text, starts = _flat(path)
    m = re.search(pattern.replace(" ", r"\s+"), text)
    if not m:
        raise LG.LedgerError(f"wording left the draft: {pattern!r} in {LG.rel(path)}")
    first = max(i for pos, i in starts if pos <= m.start())
    last = max(i for pos, i in starts if pos <= m.end() - 1)
    lines = f"{first}" if first == last else f"{first}-{last}"
    return {"source": f"{LG.rel(path)}:{lines}", "quote": " ".join(m.group(0).split())}


def part(label: str, path: Path, pattern: str, **extra) -> dict:
    out = {"label": label, **quote(path, pattern)}
    out.update(extra)
    return out


def printed_outcomes(frozen: list) -> dict:
    ""
    basis = quote(D6_VERDICTS, V6_OUTCOMES_SENTENCE)
    missing = [f for f in frozen if f not in V6_OUTCOME_NAME]
    if missing:
        raise LG.LedgerError(f"no v6 name declared for the outcome(s) {missing}")
    printed = [V6_OUTCOME_NAME[f] for f in frozen]
    for name in printed:
        if name.lower() not in basis["quote"]:
            raise LG.LedgerError(f"the v6 outcome name {name!r} is not in the sentence of {basis['source']}: "
                                 f"{basis['quote']!r}")
    return {"forms_printed": printed, "forms_printed_basis": basis}


def main():
    dc = decision_classes()
    estimates = quote(TOOLCHAIN, r"latency, energy, area and lifetime are estimates")
    names = re.match(r"(\w+), (\w+), (\w+) and (\w+) are estimates", estimates["quote"]).groups()
    icon_of = {"latency": "clock", "energy": "zap", "area": "grid", "lifetime": "hourglass"}
    frozen_forms = ["Certified", "Built and checked", "Refused by name"]
    data = {
        "id": NAME,
        "paths_relative_to": "the repository root (research)",
        "built_by": LG.rel(Path(__file__)),
        "message": ("A trained network and a chip description go in; a program, a verdict and a cost estimate "
                    "come out, with an exploration loop over the mapping."),
        "ledger_meta": LG.meta(),
        "values_plotted": "none: the figure is a schematic of Section III; each part cites the draft's wording",
        "inputs": {
            "network": part("Trained network", TOOLCHAIN,
                            r"takes a trained network, given as a graph of tensor operations with its weights"),
            "chip": part("Chip description", TOOLCHAIN,
                         r"A chip description lists core types .*? banded cost constants"),
        },
        "toolchain_label": "Toolchain",
        "stages": [
            part("Conversion and adaptation", TOOLCHAIN,
                 r"Adaptation fits the network to the properties that change the computation",
                 id="conversion", tags=["comp"], picture="staircase",
                 picture_labels=["Trained weights", "Quantization"],
                 also=quote(TOOLCHAIN, r"Conversion folds batch normalization into the weight layers")),
            part("Mapping", TOOLCHAIN,
                 r"Beyond capacity each segment is cut into passes",
                 id="mapping", tags=["cost", "adm"], picture="passes", picture_labels=["Pass"],
                 also=quote(TOOLCHAIN, r"Packing, passes and weight planning")),
            part("Check", THEORY,
                 r"A deployment passing both is \\emph\{certified\}: equal to \$\\iota\$ for every input",


                 id="check", tags=[], picture="equal_strips",
                 picture_labels=["Single-pass outputs", "Program outputs"]),
        ],
        "classes": {
            "comp": {"tag": "Computation keys", "role": "changing", "stage": "conversion", "keys": dc["comp"],
                     "stage_basis": [
                         quote(TOOLCHAIN, r"Adaptation fits the network to the properties that change the "
                                          r"computation")]},
            "cost": {"tag": "Cost-only keys", "role": "cost_only", "stage": "mapping", "keys": dc["cost"],
                     "stage_basis": [
                         quote(TOOLCHAIN, r"The weight planner, shared by build, exploration and comparison, "
                                          r"chooses which tiles stay programmed across passes and batches"),
                         quote(TOOLCHAIN, r"the copy count is a mapping decision"),
                         quote(THEORY, r"it \\emph\{changes only cost\} \(cores, packing, passes, weights kept "
                                       r"on chip, cost estimates or a refusal by name\)")]},
            "adm": {"tag": "Acceptance keys", "role": "admission", "stage": "mapping", "keys": dc["adm"],
                    "stage_basis": [
                        quote(THEORY, r"which programs the cost caps, write law and carry-buffer size accept"),
                        quote(TOOLCHAIN, r"writes are split by technology, each pass's divided over its declared "
                                         r"write units and ports"),
                        quote(TOOLCHAIN, r"accepted if it holds at every corner, ruled out by name if it fails at "
                                         r"every corner"),
                        quote(THEORY, r"\\emph\{rules programs out\} \(only which programs are accepted, as "
                                      r"the carry-buffer size and a write-endurance service-life floor do\)")]},
        },
        "outputs": {
            "program": part("Program", TOOLCHAIN,
                            r"returns a program for the cores and the host CPU with its check",
                            places=["Cores", "Host CPU"]),
            "verdict": part("Verdict", TOOLCHAIN,
                            r"or a refusal naming the stage and its reason",
                            forms=frozen_forms,
                            **printed_outcomes(frozen_forms),
                            form_basis=[
                                quote(THEORY, r"A deployment passing both is \\emph\{certified\}: equal to "
                                              r"\$\\iota\$ for every input"),
                                quote(THEORY, r"a deployment it does not establish is only \\emph\{built and "
                                              r"checked\}"),
                                quote(TOOLCHAIN, r"else it is refused by name")],
                            also=quote(TOOLCHAIN, r"else it is refused by name")),
            "estimate": {"label": "Cost estimate", **estimates,
                         "terms": [{"label": n.capitalize(), "icon": icon_of[n]} for n in names]},
        },
        "loop": part("Exploration loop", TOOLCHAIN,
                     r"the exploration adapts and checks a network once per setting of the properties that change "
                     r"the computation and visits each chip description at each core budget"),
        "pictograms": {
            "note": ("illustration only: the shapes of the pictures carry no value. One chip of two cores is drawn "
                     "throughout: a network of four layers, two layers per pass, two passes"),
            "network_layers": [3, 4, 4, 2],
            "chip_grid": [1, 2],
            "property_bars": {"comp": 14, "cost": 10, "adm": 12},
            "staircase_levels": 5,
            "passes": [[1, 1, 0, 0], [0, 0, 1, 1]],
            "strip_cells": [2, 0, 3, 1, 0, 2],
        },
    }
    out = F3 / "data" / f"{NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {LG.rel(out)}")
    for s in data["stages"]:
        print(f"  {s['label']}: {s['source']}")


if __name__ == "__main__":
    main()
