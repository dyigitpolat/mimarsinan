""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
F3 = HERE.parent
PAPER = F3.parent
sys.path.insert(0, str(PAPER / "figures_v2" / "scripts"))
import _ledger as L

NAME = "fig_capacity"


CHIP_LABEL = {
    "dynap_se_64x256x4": "DYNAP-SE",
    "recom_64x64_256": "ReCom",
    "neurram_like": "NeuRRAM",
    "hermes_64core": "HERMES-64",
    "puma_node (16-bit weights)": "PUMA node",
    "prime_mat_256x256": "PRIME",


    "ibm_analog_ai_512x512x34": "Analog-AI chip",
    "loihi_dense_equiv_128x1024": "Loihi",
    "xformer_pe_like": "X-Former",
    "isaac_like": "ISAAC",
    "truenorth_like": "TrueNorth",
}


FABRICATED, DESIGN_STUDY = "fabricated", "design_study"
CHIP_KIND = {
    "dynap_se_64x256x4": (FABRICATED, "moradi2017a", "measurements of a fabricated multi-core prototype chip"),
    "recom_64x64_256": (DESIGN_STUDY, "ji2018recom", "an architecture evaluated by simulation (its Table II, "
                                                    "Simulation Configuration)"),
    "neurram_like": (FABRICATED, "wan2022a", "measurements of a fabricated chip"),
    "hermes_64core": (FABRICATED, "gallo2022a", "measurements of a fabricated 64-core chip"),
    "puma_node (16-bit weights)": (DESIGN_STUDY, "ankit2019puma", "an architecture evaluated by simulation"),
    "prime_mat_256x256": (DESIGN_STUDY, "chi2016prime", "an architecture evaluated by simulation"),
    "ibm_analog_ai_512x512x34": (FABRICATED, "ambrogio2023an", "measurements of a fabricated chip (its Fig. 1c, a "
                                                               "micrograph of the chip)"),
    "loihi_dense_equiv_128x1024": (FABRICATED, "davies2018loihi", "a fabricated chip"),
    "xformer_pe_like": (DESIGN_STUDY, "sridharan2023xformer", "an architecture evaluated by simulation"),
    "isaac_like": (DESIGN_STUDY, "shafiee2016isaac", "an architecture evaluated by simulation (the paper's chip "
                                                     "table names it a design study)"),
    "truenorth_like": (FABRICATED, "akopyan2015truenorth", "a fabricated chip"),
}


def main():
    focal = L.value("capimagenet")
    if focal["form"] == "pending" or focal["value"] is None:
        raise L.LedgerError("capimagenet is pending: the figure has no count to print")
    led = {s["json_path"]: s["value"] for s in L.sources("capimagenet")}
    n_exceed, n_pairs = int(led["probe_estimate/exceed"]), int(led["probe_estimate/pairs"])
    if focal["display"] != f"{n_exceed} of {n_pairs}":
        raise L.LedgerError(f"capimagenet prints {focal['display']!r}, its sources hold {n_exceed} of {n_pairs}")


    chip_src = L.sources("capprobechips")
    net_src = L.sources("capinetnets")
    census_path = L.source_path(chip_src[0])
    digest = L.sha256(census_path)
    for s in chip_src + net_src:
        if L.source_path(s) != census_path or s["sha256"] != digest:
            raise L.LedgerError(f"census record differs from the ledger's source {s['path']} ({s['json_path']})")
    census = L.load_json(census_path)
    rel = L.rel(census_path)

    chip_ids = [s["value"] for s in chip_src]
    net_ids = [s["value"] for s in net_src]
    if len(chip_ids) != int(L.value("capprobechips")["value"]) or len(net_ids) != int(L.value("capinetnets")["value"]):
        raise L.LedgerError("the ledger's chip or network count differs from the names its sources list")
    if len(chip_ids) * len(net_ids) != n_pairs:
        raise L.LedgerError(f"{len(net_ids)} networks x {len(chip_ids)} chips is not the ledger's {n_pairs} pairs")

    chip_at = {c["chip"]: (i, c) for i, c in enumerate(census["chips"])}
    net_at = {n["network"]: (i, n) for i, n in enumerate(census["networks"])}
    pair_at = {(p["network"], p["chip"]): (i, p) for i, p in enumerate(census["pairs"])}

    chips = []
    for cid in chip_ids:
        i, c = chip_at[cid]
        kind, work, basis = CHIP_KIND[cid]
        chips.append({"id": cid, "label": CHIP_LABEL[cid], "kind": kind, "kind_work": work, "kind_basis": basis,
                      "capacity_cells": c["cells"], "cores": c["cores"],
                      "weight_bits": c["weight_bits"], "technology": c["technology"],
                      "eligibility": c["eligibility"], "source": f"{rel} chips/{i}/cells"})
    nets = []
    for nid in net_ids:
        i, n = net_at[nid]
        if n["dataset"] != "ImageNet":
            raise L.LedgerError(f"{nid} is not an ImageNet-class network in the census")
        nets.append({"id": nid, "label": nid, "weights": n["weights"], "source": f"{rel} networks/{i}/weights"})

    chips.sort(key=lambda c: c["capacity_cells"])
    nets.sort(key=lambda n: -n["weights"])

    ratio, ratio_source, recount = [], [], 0
    for n in nets:
        row, src = [], []
        for c in chips:
            i, p = pair_at[(n["id"], c["id"])]
            x = float(p["x_fluid"])
            if abs(x - n["weights"] / c["capacity_cells"]) > 1e-9 * max(1.0, x):
                raise L.LedgerError(f"pair {n['id']} on {c['id']}: x_fluid is not weights over cells")
            row.append(x)
            src.append(f"{rel} pairs/{i}/x_fluid")
            recount += x > 1.0
        ratio.append(row)
        ratio_source.append(src)
    if recount != n_exceed:
        raise L.LedgerError(f"the census pairs hold {recount} above 1, the ledger prints {n_exceed}")

    data = {
        "name": NAME,
        "built": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "message": f"Trained networks exceed the capacity of published chips ({focal['display']} pairs); executing "
                   "in passes that rewrite the cores closes the gap with identical outputs.",
        "focal": {**focal, "exceed": n_exceed, "pairs": n_pairs,
                  "sources": [{"path": s["path"], "json_path": s["json_path"], "value": s["value"]}
                              for s in L.sources("capimagenet")]},
        "definition": {"ratio": "weight count of the network over the weight cells of the chip description "
                                "(fluid bound, fill 1.0)",
                       "packing": census["assumptions"]["packing"],
                       "source": f"{rel} assumptions/packing"},
        "record": {"path": rel, "sha256": digest},
        "network_key": "capinetnets", "chip_key": "capprobechips",
        "chip_kind": {"values": [FABRICATED, DESIGN_STUDY],
                      "counts": {k: sum(c["kind"] == k for c in chips) for k in (FABRICATED, DESIGN_STUDY)},
                      "source": "scripts/build_fig_capacity_data.py CHIP_KIND: per column, the cited work and what "
                                "it reports (a fabricated chip's measurements, or an architecture evaluated by "
                                "simulation only). The census record and the ledger hold no such field."},
        "networks": nets, "chips": chips,
        "ratio": ratio, "ratio_source": ratio_source,
        "recount": {"exceed": recount, "pairs": len(nets) * len(chips), "agrees_with_ledger": True},
        "schematic": {"panel": "b", "values": "none (a schematic of passes on the same cores; no number drawn)"},
        "data_gaps": [],
    }
    path = L.write(NAME, data, F3 / "data")
    print(f"wrote {L.rel(path)}: {len(nets)} networks x {len(chips)} chips, {recount} of {len(nets) * len(chips)} "
          f"above 1 (ledger: {focal['display']}, {focal['class']}, {focal['form']})")


if __name__ == "__main__":
    main()
