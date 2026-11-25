""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
sys.path.insert(0, str(F3.parent / "figures_v2" / "scripts"))
import _ledger as LG
import _names as NM

NAME = "fig_chipprops"
IDENTITY_BITS = 32
GRID_MEMBERS = ("mvm_int_readout_lenet5", "t0_43v")
W34_NODE = re.compile(r"^members/(?P<member>[^/]+)/readings/loss_pp/")
CIFAR_NODE = re.compile(r"^cells/\d+/")
X_MAX = 30.0
X_MINUS_ONE_MIN = 1.0e-6
N_SAMPLES = 240


X_LABEL_PARTS = [["Weight overflow", "text"], ["x", "symbol"], ["w", "sub"]]


def src(path: Path, node: str) -> str:
    return f"{LG.rel(path)}#{node} (sha256 {LG.sha256(path)[:16]})"


def first_number(text: str) -> float:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?", str(text))
    if not m:
        raise LG.LedgerError(f"no number in {text!r}")
    return float(m.group(0).replace(",", ""))


def numbers(text: str) -> list:
    return [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?", str(text))]


def check(what: str, got: float, want: float, tol: float):
    if abs(got - want) > tol:
        raise LG.LedgerError(f"{what}: the record gives {got!r}, the ledger prints {want!r}")


def grid_curves() -> list:
    path = LG.record("adcgrid", "summary.json")
    summ = json.loads(path.read_text())
    split = LG.value("adclenetfour")
    if LG.names_test_split(LG.entry("adclenetfour")):
        raise LG.LedgerError("adclenetfour counts a test split: the grid is not drawn")
    m = re.search(r"of ([\d,]+) images", split["display"])
    n = int(m.group(1).replace(",", ""))
    out = []
    for mid in GRID_MEMBERS:
        rec = summ["members"][mid]
        per_arm = {}
        for arm in ("untrained", "tuned"):
            by_bits = {}
            for k, c in rec[arm]["counts"].items():
                bits = int(k.split("/")[1].replace("adc", ""))
                by_bits.setdefault(bits, set()).add(c)
            assert all(len(cs) == 1 for cs in by_bits.values()), (mid, arm)
            per_arm[arm] = {b: next(iter(cs)) for b, cs in sorted(by_bits.items())}
        assert per_arm["untrained"] == per_arm["tuned"], mid
        counts = per_arm["untrained"]
        ident = counts[IDENTITY_BITS]
        pts = [{"bits": b, "loss_pp": 100.0 * (ident - c) / n, "count": c}
               for b, c in counts.items() if b != IDENTITY_BITS]
        out.append({"id": mid, "network": NM.member_name(mid), "dataset": "MNIST", "images": n,
                    "identity_count": ident, "status": "measured", "points": pts,
                    "source": src(path, f"members/{mid}/untrained/counts (of {n:,} images, ledger adclenetfour)")})
    return out


def cifar10_curve() -> dict:
    for s in LG.sources("adccifarten"):
        if not CIFAR_NODE.match(str(s.get("json_path", ""))):
            continue
        path = LG.source_path(s)
        rec = LG.load_json(path)
        n = int(rec["count_split"])
        by = {}
        for c in rec["cells"]:
            by.setdefault(int(c["adc_bits"]), []).append(float(c["mean"]))
        ident = by[IDENTITY_BITS][0]
        pts = [{"bits": b, "loss_pp": 100.0 * (ident - v[0]) / n} for b, v in sorted(by.items())
               if b != IDENTITY_BITS]
        return {"id": "resnet20_cifar10", "network": "ResNet-20", "dataset": "CIFAR-10", "images": n,
                "status": "measured", "points": pts, "key": "adccifarten",
                "source": src(path, "cells[*].mean (mean of the seeds, against adc_bits 32)")}
    raise LG.LedgerError("adccifarten: no cells source")


def cifar100_curve() -> dict:
    for s in LG.sources("adclosscifar"):
        m = W34_NODE.match(str(s.get("json_path", "")))
        if not m:
            continue
        path = LG.source_path(s)
        rec = LG.load_json(path)
        if rec.get("format") != "w34-collect/1":
            raise LG.LedgerError(f"adclosscifar: record format {rec.get('format')!r}")
        mem = rec["members"][m.group("member")]
        loss = {b: {int(a): float(v) for a, v in c.items()} for b, c in mem["readings"]["loss_pp"].items()}
        budgets = sorted(loss)
        first = loss[budgets[0]]
        assert all(loss[b] == first for b in budgets), "the loss differs across core budgets"
        pts = [{"bits": a, "loss_pp": v} for a, v in sorted(first.items()) if a != IDENTITY_BITS]
        return {"id": "resnet18_cifar100", "network": "ResNet-18", "dataset": "CIFAR-100",
                "images": mem.get("images"), "status": "measured", "points": pts, "key": "adclosscifar",
                "training": "without the converter",
                "source": src(path, f"members/{m.group('member')}/readings/loss_pp (equal at budgets "
                                    f"{', '.join(budgets)})")}
    raise LG.LedgerError("adclosscifar: no loss_pp source")


def converter() -> dict:
    lenet, cnn = grid_curves()
    lenet["key"], cnn["key"] = "adclenetfour", "adctzerofour"
    lenet["point_keys"] = {4: "adclenetfour", 3: "adclenetthree"}
    c10, c100 = cifar10_curve(), cifar100_curve()
    at = lambda s, b: next(p["loss_pp"] for p in s["points"] if p["bits"] == b)
    check("LeNet-5 at 4 bits", at(lenet, 4), first_number(LG.value("adclenetfour")["value"]), 0.006)
    check("LeNet-5 at 3 bits", at(lenet, 3), first_number(LG.value("adclenetthree")["value"]), 0.006)
    check("8-layer CNN at 4 bits", at(cnn, 4), first_number(LG.value("adctzerofour")["value"]), 0.006)
    check("ResNet-20 at 4 bits", at(c10, 4), first_number(LG.value("adccifarten")["value"]), 0.006)
    check("ResNet-18 at 4 bits", at(c100, 4), first_number(LG.value("adclosscifar")["value"]), 0.006)
    series = [c100, c10, cnn, lenet]
    for s in series:
        s["label"] = f"{s['network']}, {s['dataset']}"
        printed = s.pop("point_keys", {4: s["key"]})
        for p in s["points"]:
            p["key"] = printed.get(p["bits"])
            if p["key"] is None:
                p["checked_against"] = s["key"]
            else:
                check(f"{s['label']} at {p['bits']} bits", p["loss_pp"],
                      first_number(LG.value(p["key"])["value"]), 0.006)
    bits = sorted({p["bits"] for s in series for p in s["points"]}, reverse=True)
    aware = LG.value("adcarmrest")
    gaps = []
    if aware["form"] == "pending":
        gaps.append({"what": "ResNet-18 on CIFAR-100 with the converter in training: its loss curve",
                     "key": "adcarmrest", "form": aware["form"], "display": aware["display"]})
    if 3 not in {p["bits"] for p in c100["points"]}:
        gaps.append({"what": "ResNet-18 on CIFAR-100 at 3 converter bits: not in its run record (4, 5, 6 and "
                             "8 bits read)", "key": "adclosscifar"})
    return {"x_label": "Converter bits", "y_label": "Accuracy loss (pp)", "title": "Converter precision",
            "bits": bits, "series": series, "gaps": gaps,
            "ledger": LG.values("adcgrid", "spreadbudgets", "adclenetfour", "adclenetthree", "adctzerofour",
                                "adccifarten", "adclosscifar", "adcarmpair", "adcarmrest")}


def band_ends(v: dict) -> list:
    txt = str(v["value"]) + " " + str(v["display"])
    ends = [float(m) for m in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?e[+-]?\d+)", txt)]
    ends += [10.0 ** int(m) for m in re.findall(r"10\^\{?(\d+)\}?", txt)]
    ends = sorted(set(ends))
    if len(ends) < 2:
        raise LG.LedgerError(f"{v['key']}: no band in {v['display']!r}")
    return [ends[0], ends[-1]]


def pow10_label(v: float) -> str:
    e = math.log10(v)
    if abs(e - round(e)) > 1e-9:
        raise LG.LedgerError(f"band end {v!r} is not a power of ten")
    return rf"$10^{{{round(e)}}}$"


def endurance() -> dict:
    rate = first_number(LG.value("rate")["value"])
    life_s = first_number(LG.value("lifefloor")["value"]) * 3600.0
    pulses = first_number(LG.value("rrampulses")["value"])
    low, high = band_ends(LG.value("rramendurance"))
    x_anchor = first_number(LG.value("inetxfluid")["value"])
    k_bw = first_number(LG.value("kminbandwidth")["value"])

    def per_overflow(cycles: float) -> float:
        return rate * life_s * pulses / cycles

    lo_e, hi_e = math.log10(X_MINUS_ONE_MIN), math.log10(X_MAX - 1.0)
    xs = [1.0 + 10.0 ** (lo_e + (hi_e - lo_e) * i / (N_SAMPLES - 1)) for i in range(N_SAMPLES)]
    xs[-1] = X_MAX
    printed = numbers(LG.value("kminendurance")["value"])
    limit = numbers(LG.value("batchonelimit")["value"])
    corners = []
    for cycles, want_k, want_x in ((low, printed[1], limit[1]), (high, printed[0], limit[0])):
        c = per_overflow(cycles)
        k_anchor = c * (x_anchor - 1.0)
        x_one = 1.0 + 1.0 / c
        check(f"minimum batch at {cycles:g} cycles", math.ceil(k_anchor), want_k, 0.0)


        check(f"overflow at batch 1, {cycles:g} cycles", math.ceil(x_one * 100 - 1e-9) / 100, want_x, 0.006)
        corners.append({
            "cycles": cycles, "label": rf"Endurance {pow10_label(cycles)} cycles",
            "per_overflow": c, "x": xs, "k": [c * (x - 1.0) for x in xs],
            "anchor": {"k": k_anchor, "batch": math.ceil(k_anchor), "status": "estimated", "key": "kminendurance",
                       "source": f"closed form (x - 1) r T p / E at x = ledger inetxfluid, E = {cycles:g} cycles "
                                 f"(ledger rramendurance), r = ledger rate, T = ledger lifefloor, p = ledger "
                                 f"rrampulses; its ceiling equals ledger kminendurance"},
            "x_at_batch_one": {"value": x_one, "key": "batchonelimit"},
            "source": "closed form (x - 1) r T p / E from ledger rate, lifefloor, rrampulses and rramendurance",
        })
    bw = {"key": "kminbandwidth", "label": "Batch for declared rate", "status": "estimated",
          "role": "reference: it bears on time per inference and the sustainable rate and rules nothing out "
                  "(the ledger's words for kminbandwidth)",
          "anchor": {"k": k_bw, "display": LG.value("kminbandwidth")["display"]},
          "x": xs, "k": [k_bw * (x - 1.0) / (x_anchor - 1.0) for x in xs],
          "source": "ledger kminbandwidth at x = ledger inetxfluid; the line is proportional to x - 1 (a period's "
                    "writes grow with the weights beyond capacity), as the closed form K >= r t_w"}
    lo_b, hi_b = corners[1]["anchor"]["batch"], corners[0]["anchor"]["batch"]
    return {"x_label": "Weight overflow x_w", "x_label_parts": X_LABEL_PARTS,
            "y_label": "Minimum batch", "title": "Write endurance",
            "formula": "K = (x - 1) r T p / E", "corner_heading": "Closed form",
            "constants": {"rate_per_s": {"value": rate, "key": "rate"},
                          "life_s": {"value": life_s, "key": "lifefloor"},
                          "pulses_per_write": {"value": pulses, "key": "rrampulses"},
                          "band_cycles": {"value": [low, high], "key": "rramendurance"}},
            "x_max": X_MAX, "corners": corners, "bandwidth": bw,
            "anchor": {"label": "ResNet-50 on NeuRRAM", "x": x_anchor, "key": "inetxfluid",
                       "focal": f"Batch {lo_b} to {hi_b}"},
            "not_drawn": {
                "write_law_instance": {"why": "the write law's verdict on a packed instance (its most-written cell) "
                                              "lies above the closed form, a lower bound; printed in the text",
                                       "value": LG.value("kstarneurram")},
                "crossover": {"why": "printed in the text", "value": LG.value("ecrossneurram")},
            },
            "ledger": LG.values("rate", "lifefloor", "rrampulses", "rramendurance", "inetxfluid", "kminendurance",
                                "kminbandwidth", "batchonelimit", "neurramengines")}


def main():
    data = {"figure": NAME, "label": "fig:chipprops", "section": "VI-D",
            "message": "Converter precision changes accuracy, and write endurance sets a minimum batch that the "
                       "closed form predicts.",
            "converter": converter(), "endurance": endurance(),
            "caption_values": LG.values("rate", "inetxfluid")}
    out = LG.write(NAME, data, F3 / "data")
    conv, end = data["converter"], data["endurance"]
    print(f"wrote {out}")
    for s in conv["series"]:
        print("  (a)", s["label"], {p["bits"]: round(p["loss_pp"], 2) for p in s["points"]})
    for g in conv["gaps"]:
        print("  (a) gap:", g["what"])
    for c in end["corners"]:
        print(f"  (b) {c['cycles']:g} cycles: K per unit overflow {c['per_overflow']:.4f}, at x "
              f"{end['anchor']['x']}: {c['anchor']['k']:.3f} (batch {c['anchor']['batch']}), batch 1 to x "
              f"{c['x_at_batch_one']['value']:.3f}")
    print(f"  (b) bandwidth at the anchor: {end['bandwidth']['anchor']['k']:,.0f}")


if __name__ == "__main__":
    main()
