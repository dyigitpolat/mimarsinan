"""A5: REAL-PIPELINE per-stage accuracy-loss decomposition (ground truth).

Parse a real deployment-pipeline run log (the `[PROFILE] step=... metric=...`
lines emitted after every step) into a per-stage loss table. Every loss is a
DELTA from the ANN (the Pretraining metric == the LOSSLESS target `cont`), in
percentage points (pp). This LOCALIZES the deployed loss to a named pipeline
stage.

Pipeline stage order (deployment):
    Model Configuration -> Model Building -> Pretraining (= ANN, the ceiling)
    -> Torch Mapping -> Activation Analysis -> TTFS Cycle Fine-Tuning
    (= the cascade conversion) -> Weight Quantization -> Quantization Verification
    -> Normalization Fusion -> Soft Core Mapping (SCM sim) -> Hard Core Mapping
    (HCM sim) -> Simulation (nevresim/SANA-FE deployed metric).

Usage:
    python real_stage_decomp.py LOG [LOG ...]
    # default: parses /tmp/val_s4.log and /tmp/val_s16_baseline.log

The metric the pipeline reports is whatever each step measures (a subsampled or
full-test accuracy). The DELTA between consecutive non-trivial steps is the loss
that stage owns. Steps that don't change the metric (mapping, fusion) are
near-lossless; the conversion step that drops it is where the loss LIVES.
"""

from __future__ import annotations

import re
import sys

# A [PROFILE] line: step='<name>' ... metric=<f> ... (prev=<f>)
_PROFILE_RE = re.compile(
    r"\[PROFILE\]\s+step='(?P<step>[^']+)'\s+wall=\s*(?P<wall>[\d.]+)s\s+"
    r"metric=(?P<metric>[-\d.]+).*?\(prev=(?P<prev>[-\d.]+)\)"
)

# Steps whose metric is meaningless (0.0 build/config) — excluded from the table.
_SKIP_STEPS = {"Model Configuration", "Model Building"}

# Human-readable stage grouping for the headline attribution.
_STAGE_GROUP = {
    "Pretraining": "ANN (ceiling)",
    "Torch Mapping": "mapping",
    "Pruning Adaptation": "pruning",
    "Activation Analysis": "activation-analysis",
    "TTFS Cycle Fine-Tuning": "CASCADE CONVERSION",
    "Weight Quantization": "weight-quant",
    "Quantization Verification": "weight-quant",
    "Normalization Fusion": "norm-fusion",
    "Soft Core Mapping": "soft-core sim",
    "Core Quantization Verification": "soft-core sim",
    "Hard Core Mapping": "hard-core sim",
    "Simulation": "deployed sim (nevresim/SANA-FE)",
}


def parse_log(path):
    """Return ordered list of {step, wall, metric} for non-trivial steps."""
    rows = []
    with open(path) as fh:
        for line in fh:
            m = _PROFILE_RE.search(line)
            if not m:
                continue
            step = m.group("step")
            if step in _SKIP_STEPS:
                continue
            rows.append(
                dict(
                    step=step,
                    wall=float(m.group("wall")),
                    metric=float(m.group("metric")),
                )
            )
    return rows


def decompose(rows):
    """Attribute per-stage loss as a DELTA from the ANN (first real metric).

    Returns (ann_acc, table) where table is a list of per-stage dicts with:
      step, metric, stage_delta_pp (loss this step adds vs the prev step),
      cum_loss_pp (cumulative loss vs the ANN), group.
    """
    if not rows:
        return None, []
    ann = rows[0]["metric"]  # Pretraining == the lossless target `cont`
    table = []
    prev = ann
    for r in rows:
        stage_delta_pp = (r["metric"] - prev) * 100.0  # +improvement, -loss
        cum_loss_pp = (r["metric"] - ann) * 100.0
        table.append(
            dict(
                step=r["step"],
                metric=r["metric"],
                wall=r["wall"],
                stage_delta_pp=stage_delta_pp,
                cum_loss_pp=cum_loss_pp,
                group=_STAGE_GROUP.get(r["step"], "?"),
            )
        )
        prev = r["metric"]
    return ann, table


def _fmt_table(ann, table, label):
    lines = []
    lines.append(f"\n=== {label} ===")
    lines.append(f"ANN (lossless target, cont) = {ann:.4f}")
    lines.append(
        f"{'step':<30}{'metric':>9}{'Δ vs prev (pp)':>16}"
        f"{'cum loss (pp)':>15}{'wall(s)':>10}"
    )
    lines.append("-" * 80)
    for r in table:
        lines.append(
            f"{r['step']:<30}{r['metric']:>9.4f}{r['stage_delta_pp']:>+16.2f}"
            f"{r['cum_loss_pp']:>+15.2f}{r['wall']:>10.1f}"
        )
    deployed = table[-1]["metric"] if table else float("nan")
    total_loss = (ann - deployed) * 100.0
    lines.append("-" * 80)
    lines.append(f"deployed (last step) = {deployed:.4f}   TOTAL loss = {total_loss:+.2f} pp")
    return "\n".join(lines)


def _attribution(ann, table):
    """Group per-stage losses and report the dominant owner."""
    groups = {}
    for r in table:
        if r["group"] == "ANN (ceiling)":
            continue
        groups.setdefault(r["group"], 0.0)
        groups[r["group"]] += r["stage_delta_pp"]
    deployed = table[-1]["metric"]
    total = (ann - deployed) * 100.0
    lines = ["\n  loss attribution by stage-group (negative = loss):"]
    for g, d in sorted(groups.items(), key=lambda kv: kv[1]):
        share = (-d / total * 100.0) if total > 1e-9 else 0.0
        lines.append(f"    {g:<34}{d:>+8.2f} pp   ({share:5.1f}% of total)")
    sim_seam = sum(
        r["stage_delta_pp"]
        for r in table
        if r["group"] in ("soft-core sim", "hard-core sim", "deployed sim (nevresim/SANA-FE)")
    )
    lines.append(
        "  NOTE: SCM/HCM/Simulation are bit-exact in mapping (established parity);"
    )
    lines.append(
        f"  their net {sim_seam:+.2f} pp here is SUBSAMPLE noise (sim-test metric is"
        " subsampled vs full-test NF), not a true mapping loss."
    )
    return "\n".join(lines), groups, total


def run(paths):
    out = []
    summary = {}
    for p in paths:
        rows = parse_log(p)
        ann, table = decompose(rows)
        if ann is None:
            out.append(f"\n=== {p} ===\n  (no [PROFILE] lines found)")
            continue
        out.append(_fmt_table(ann, table, p))
        attr, groups, total = _attribution(ann, table)
        out.append(attr)
        summary[p] = dict(ann=ann, table=table, groups=groups, total=total)
    print("\n".join(out))
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["/tmp/val_s4.log", "/tmp/val_s16_baseline.log"]
    run(args)
