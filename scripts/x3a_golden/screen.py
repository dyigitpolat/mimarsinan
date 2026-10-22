"""The X3a model-level screen: instances per arm, counterexamples, Clopper-Pearson bounds, coverage.

One instance (``run_instance``): generate (program, declaration, inputs, knobs) from the (arm, seed)
string; run the reference (identity semantics) and the identity packing on the machine, which must agree
(a golden self-check: a disagreement raises ``GoldenDefect``); pack at random under the arm's switch,
validate the artifact against every hypothesis that is on, run it, and compare per neuron and window
with the identity program (relation ``packed``); compare the reference with the integer network where
the box claims it and its checks admit it (relation ``network``). A counterexample is a mismatch in
either relation of an admitted instance; refusals are named and counted, never compared.

Run (from ``scripts/``): ``python -m x3a_golden.screen --n-on 1000 --n-off 100 --workers 2 --out F``.
The JSON's ``deterministic`` part is a pure function of the seeds (``--twice`` reruns a fraction and
asserts byte identity); walls sit apart under ``walls``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .generator import CONTROLS, generate
from .hypotheses import BY_LABEL, D17_ARMS, LABELS, admit_network, exercised, validate
from .machine import run_packed
from .packer import GoldenRefusal, coalesce, fragments
from .program import canonical_json
from .reference import run_network, run_reference
from .schedule import Packed, identity_pack, schedule


class GoldenDefect(AssertionError):
    """The golden disagrees with itself (reference against the identity packing): never a counterexample."""


def pack(prog, rng: random.Random, knobs, off: FrozenSet[str], batch: int) -> Packed:
    """A random packing; a capacity or budget refusal is redrawn three times, then once with no optional
    fusion, split, coalescing or pins, and only then refused by name."""
    last: Optional[GoldenRefusal] = None
    for attempt in range(4):
        k = knobs if attempt < 3 else dict(knobs, fuse=0.0, split=0.0, coalesce=0.0, pin=0.0, cut=0.0)
        try:
            cores = coalesce(prog, fragments(prog, rng, k), rng, k, off)
            return schedule(prog, cores, rng, k, off, batch)
        except GoldenRefusal as err:
            if err.name not in ("CapacityRefused", "PassBudgetRefused"):
                raise
            last = err
    assert last is not None
    raise last


def divergence(prog, expect: Dict[int, np.ndarray], got: Dict[int, np.ndarray]) -> Optional[Dict[str, int]]:
    """The first (node, neuron, sample) where two runs differ, in node order, or None."""
    for node in prog.nodes:
        if node.kind == "input":
            continue
        diff = np.argwhere(expect[node.nid] != got[node.nid])
        if diff.size:
            s, n = (int(v) for v in diff[0])
            return {"node": node.nid, "neuron": n, "sample": s, "expected": int(expect[node.nid][s, n]),
                    "got": int(got[node.nid][s, n]), "count": int(len(diff))}
    return None


def _net_divergence(prog, inputs, ref) -> Optional[Dict[str, int]]:
    net = run_network(prog, inputs)
    assert net is not None, "relation network is claimed only where run_network evaluates N"
    return divergence(prog, net, ref)


def _decisions(packed: Packed, stats) -> Dict[str, int]:
    cores = packed.cores
    firing: Dict[int, int] = {}
    for c in cores:
        for lay in {lay for lay, _, _ in c.slots} if not c.partial else ():
            firing[lay] = firing.get(lay, 0) + 1
    per_seg: Dict[int, int] = {}
    for seg, _ in packed.passes:
        per_seg[seg] = per_seg.get(seg, 0) + 1
    return {"cores": len(cores), "passes": len(packed.passes),
            "multipass_segments": sum(v > 1 for v in per_seg.values()),
            "fused_slots": sum(g >= 0 for c in cores for _, _, g in c.slots),
            "partial_cores": sum(c.partial for c in cores),
            "coalesced_cores": sum(len({lay for lay, _, _ in c.slots}) > 1 for c in cores),
            "split_layers": sum(v > 1 for v in firing.values()),
            "permuted_axons": sum(c.axons != sorted(c.axons) for c in cores),
            "pinned": len(packed.pinned), "batch_K": packed.batch_K, "carried_reads": stats["carried_reads"],
            "writes": stats["writes"], "stale_runs": stats["stale_runs"]}


def _declared(prog) -> Dict[str, Any]:
    """The declared variables of an instance, for the coverage report (one value per variable)."""
    d, longest = prog.decl, {}
    for node in prog.nodes:
        srcs = [longest[s] for s in node.sources]
        longest[node.nid] = (max(srcs, default=0)) + (node.kind == "neural")
    return {"T": d.T, "comparator": d.comparator, "membrane_bits": d.membrane_bits, "core_types": len(d.core_types),
            "cores_declared": sum(ct.count for ct in d.core_types), "pass_budget": d.pass_budget,
            "accumulator": d.accumulator, "depth": max(longest.values()),
            "max_width": max(n.size for n in prog.nodes if n.kind == "neural")}


def run_instance(arm: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"x3a-golden/{arm}/{seed}")
    prog, inputs, knobs, motifs = generate(rng, arm)
    label = arm.split("@")[0]
    off = frozenset() if arm == "all_on" else frozenset({label})
    d = prog.decl
    rec: Dict[str, Any] = {"arm": arm, "seed": seed, "program": prog.digest()[:16], "semantics": d.semantics,
                           "granularity": d.granularity, "reset": d.reset, "motifs": motifs,
                           "declared": _declared(prog)}
    B = next(iter(inputs.values())).shape[0]
    ref, diag = run_reference(prog, inputs)
    ident, _ = run_packed(prog, identity_pack(prog, B), inputs, exact=True)
    bad = divergence(prog, ref, ident)
    if bad is not None:
        raise GoldenDefect(f"{arm}/{seed}: reference and identity packing differ at {bad}")
    rec["nonzero_outputs"] = int(sum((ref[n.nid] != 0).sum() for n in prog.nodes if n.kind != "input"))
    claimed, refused_net = admit_network(prog, diag, off)
    rec["network"] = {"claimed": claimed, "refused_by": refused_net,
                      "divergence": _net_divergence(prog, inputs, ref) if claimed else None}
    packed, stats = None, None
    try:
        packed = pack(prog, rng, knobs, off, B)
    except GoldenRefusal as err:
        rec["packed"] = {"refused_by": err.name}
    else:
        violated = validate(prog, packed, off)
        if violated:
            rec["packed"] = {"refused_by": "+".join(violated)}
        else:
            out, stats = run_packed(prog, packed, inputs, off)
            rec["packed"] = {"refused_by": None, "divergence": divergence(prog, ident, out),
                             "decisions": _decisions(packed, stats)}
    rec["exercised"] = exercised(label, prog, packed, stats, diag) if arm != "all_on" else None
    net_cex = rec["network"]["divergence"] is not None
    pk_cex = rec["packed"].get("divergence") is not None
    rec["counterexample"] = net_cex or pk_cex
    return rec


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (1 - alpha) interval for k of n; at k = 0 the upper end is the one-sided bound 1 - alpha**(1/n)."""
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        return 0.0, 1 - alpha ** (1 / n)

    def cdf(p: float, kk: int) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 1.0 if kk >= n else 0.0
        lp, lq = math.log(p), math.log1p(-p)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp
                            + (n - i) * lq) for i in range(kk + 1))

    def root(f, target: float, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = root(lambda p: 1 - cdf(p, k - 1), alpha / 2, True)
    upper = 1.0 if k == n else root(lambda p: cdf(p, k), alpha / 2, False)
    return lower, upper


def _job(args: Tuple[str, int]) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    rec = run_instance(*args)
    return rec, time.perf_counter() - t0


def summarize(arm: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    packed = [r for r in recs if r["packed"].get("refused_by") is None]
    network = [r for r in recs if r["network"]["claimed"]]
    cex = [r for r in recs if r["counterexample"]]
    row: Dict[str, Any] = {
        "instances": len(recs), "packed_compared": len(packed), "network_compared": len(network),
        "counterexamples": len(cex),
        "packed_counterexamples": sum(r["packed"].get("divergence") is not None for r in recs),
        "network_counterexamples": sum(r["network"]["divergence"] is not None for r in recs),
        "first_counterexamples": [{"seed": r["seed"], "semantics": r["semantics"],
                                   "packed": r["packed"].get("divergence"), "network": r["network"]["divergence"]}
                                  for r in cex[:3]],
        "refusals": {}, "semantics": {}, "motifs": {}, "decisions": {}, "counterexamples_by_semantics": {},
        "declared": {},
        "degenerate": sum(r["nonzero_outputs"] == 0 for r in recs)}
    for r in recs:
        for key in [r["packed"].get("refused_by")] + r["network"]["refused_by"]:
            if key:
                row["refusals"][key] = row["refusals"].get(key, 0) + 1
        row["semantics"][r["semantics"]] = row["semantics"].get(r["semantics"], 0) + 1
        if r["counterexample"]:
            cbs = row["counterexamples_by_semantics"]
            cbs[r["semantics"]] = cbs.get(r["semantics"], 0) + 1
        for m, v in r["motifs"].items():
            row["motifs"][m] = row["motifs"].get(m, 0) + (v > 0)
        for var, v in r["declared"].items():
            cell = row["declared"].setdefault(var, {})
            cell[str(v)] = cell.get(str(v), 0) + 1
        for k, v in r["packed"].get("decisions", {}).items():
            row["decisions"][k] = row["decisions"].get(k, 0) + (v > 1 if k == "batch_K" else v > 0)
    if arm == "all_on":
        row["packed_upper95_one_sided"] = clopper_pearson(row["packed_counterexamples"], len(packed))[1]
        row["network_upper95_one_sided"] = clopper_pearson(row["network_counterexamples"], len(network))[1]
    else:
        ex = [r for r in recs if r["exercised"]]
        k = sum(r["counterexample"] for r in ex)
        row.update(exercising=len(ex), counterexamples_exercising=k, rate_ci95=clopper_pearson(k, len(ex)),
                   teeth=len(cex) > 0, hypothesis=BY_LABEL[arm.split("@")[0]].name)
    return row


def screen(arms: List[str], n_on: int, n_off: int, workers: int, twice: float = 0.0, seed0: int = 0) -> Dict[str, Any]:
    jobs = [(arm, s) for arm in arms for s in range(seed0, seed0 + (n_on if arm == "all_on" else n_off))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(_job, jobs, chunksize=4))
    recs = {arm: [r for r, _ in done if r["arm"] == arm] for arm in arms}
    walls = {arm: sorted(w for r, w in done if r["arm"] == arm) for arm in arms}
    rows = {arm: summarize(arm, recs[arm]) for arm in arms}
    det = {"arms": rows, "records_sha256": hashlib.sha256(canonical_json(
        [r for r, _ in done]).encode()).hexdigest()}
    off_arms = [a for a in arms if a in D17_ARMS]
    det["d17_arms_with_teeth"] = sum(rows[a]["teeth"] for a in off_arms)
    det["d17_arms_run"] = len(off_arms)
    if twice:
        step = max(1, int(round(1 / twice)))
        again = [run_instance(*job) for job in jobs[::step]]
        first = {(r["arm"], r["seed"]): r for r, _ in done}
        det["twice"] = {"rerun": len(again), "identical": all(canonical_json(r) == canonical_json(
            first[(r["arm"], r["seed"])]) for r in again)}
    med = {a: (w[len(w) // 2] if w else None) for a, w in walls.items()}
    return {"deterministic": det, "walls": {"median_s": med, "total_s": sum(sum(w) for w in walls.values())}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="all", help="comma list of arms, 'all' (all_on and every label) or "
                                                  "'controls' (the LABEL@variant negative controls)")
    ap.add_argument("--n-on", type=int, default=1000)
    ap.add_argument("--n-off", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--twice", type=float, default=0.01, help="fraction of instances rerun for byte identity")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (screen seeds start at 0, plan D20)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 workers (plan section 10.2)")
    arms = ["all_on", *LABELS] if args.arms == "all" else sorted(CONTROLS) if args.arms == "controls" \
        else args.arms.split(",")
    result = screen(arms, args.n_on, args.n_off, args.workers, args.twice, args.seed0)
    result["deterministic"]["invocation"] = {"arms": arms, "n_on": args.n_on, "n_off": args.n_off,
                                             "seed0": args.seed0, "twice": args.twice}
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(result) + "\n")
    det = result["deterministic"]
    keep = ("instances", "packed_compared", "network_compared", "counterexamples", "exercising", "teeth",
            "packed_upper95_one_sided", "network_upper95_one_sided")
    print(canonical_json({a: {k: v for k, v in r.items() if k in keep}
                          for a, r in det["arms"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
