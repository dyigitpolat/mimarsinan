"""Unit tests for the publication experiment-matrix harness.

Covers the three batch generators (F1 CIs+ablations, F2 baseline head-to-head,
F3 dual-regime) against the live backlog schema (the dict shape
``scheduler.instantiate`` consumes), and the three aggregators (mean+/-CI,
head-to-head delta, dual-regime delta) against a SYNTHETIC ledger.

Self-contained: bootstraps ``scripts/campaign`` and ``scripts/gpu`` onto
``sys.path`` so ``from experiment_matrix import ...`` and the reused
``scheduler.instantiate`` resolve under the project's bare ``pytest`` command
(PYTHONPATH=src:spikingjelly only).
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "scripts", "gpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_matrix as em  # noqa: E402
import scheduler as sch  # noqa: E402


def test_default_matrix_templates_exist():
    """Every template ``default_matrix()`` ships MUST exist on disk — else the
    primary deep_cnn studies silently never enqueue (``scheduler.refill`` skips
    a FileNotFoundError batch, so the bug is non-fatal but the studies vanish)."""
    cells = em.default_matrix()
    assert cells, "default_matrix() is empty"
    for cell in cells:
        path = os.path.join(_REPO, cell.template)
        assert os.path.exists(path), (
            f"default_matrix template missing: {cell.template!r} "
            f"(resolved {path!r}) — the study would never instantiate"
        )


# ---------------------------------------------------------------------------
# Cell fixtures: (model, dataset, template) covered/valid points.
# ---------------------------------------------------------------------------
F_CELLS = [
    em.MatrixCell(model="deep_cnn", dataset="MNIST_DataProvider",
                  template="templates/mnist_deep_cnn.json"),
    em.MatrixCell(model="lenet5", dataset="FashionMNIST_DataProvider",
                  template="templates/mnist_lenet5_synchronized.json"),
]


# ---------------------------------------------------------------------------
# Schema validation: the contract every generated batch must satisfy.
# ---------------------------------------------------------------------------
def _assert_schema(batch: dict) -> None:
    # validate_batch is the SSOT; it must accept a well-formed batch silently.
    em.validate_batch(batch)
    # The required keys scheduler.instantiate / Scheduler.refill read.
    for key in ("id", "template", "grid", "id_template"):
        assert key in batch, f"missing required backlog key {key!r}"
    assert isinstance(batch["id"], str) and batch["id"]
    assert isinstance(batch["template"], str) and batch["template"]
    assert isinstance(batch["grid"], dict) and batch["grid"]
    assert isinstance(batch["id_template"], str)
    # Every grid axis is a non-empty list; id_template references only leaf names.
    leaves = set()
    for path, vals in batch["grid"].items():
        assert isinstance(vals, list) and len(vals) >= 1, path
        leaves.add(path.split(".")[-1])
    fmt_fields = {fn for _, fn, _, _ in __import__("string").Formatter().parse(
        batch["id_template"]) if fn}
    assert fmt_fields <= leaves, (
        f"id_template fields {fmt_fields - leaves} not in grid leaves {leaves}")
    # base, when present, is a flat dotted->value map (set_path-compatible).
    assert isinstance(batch.get("base", {}), dict)
    for k in batch.get("base", {}):
        assert isinstance(k, str) and k


def test_validate_batch_rejects_malformed():
    good = {"id": "x", "template": "t.json", "grid": {"seed": [0]},
            "id_template": "x_s{seed}"}
    em.validate_batch(good)  # no raise
    with pytest.raises(em.BatchSchemaError):
        em.validate_batch({"template": "t.json", "grid": {"seed": [0]},
                           "id_template": "x"})  # no id
    with pytest.raises(em.BatchSchemaError):
        em.validate_batch({"id": "x", "template": "t.json", "grid": {},
                           "id_template": "x"})  # empty grid
    with pytest.raises(em.BatchSchemaError):
        em.validate_batch({"id": "x", "template": "t.json",
                           "grid": {"seed": []}, "id_template": "x"})  # empty axis
    with pytest.raises(em.BatchSchemaError):
        # id_template references a field not present in the grid leaves
        em.validate_batch({"id": "x", "template": "t.json",
                           "grid": {"seed": [0]}, "id_template": "x_{depth}"})


# ---------------------------------------------------------------------------
# F1 — CIs + ablations: multi-seed cells.
# ---------------------------------------------------------------------------
def test_f1_batches_are_schema_valid_and_multiseed():
    batches = em.gen_f1_batches(F_CELLS, seeds=(0, 1, 2, 3, 4))
    assert batches
    for b in batches:
        _assert_schema(b)
        # F1 must vary the seed (the CI axis) with >= 2 seeds.
        seed_axes = [k for k in b["grid"] if k.split(".")[-1] == "seed"]
        assert seed_axes, f"F1 batch {b['id']} has no seed axis"
        assert len(b["grid"][seed_axes[0]]) >= 2
        assert b.get("tags", {}).get("study") == "F1"


def test_f1_batches_instantiate_distinct_ids(tmp_path):
    """Every grid point yields a distinct job id via scheduler.instantiate."""
    batches = em.gen_f1_batches(F_CELLS, seeds=(0, 1, 2))
    for b in batches:
        # instantiate reads the template file relative to REPO; point it at a
        # real stub template so the real scheduler walk runs unmodified.
        ids = _instantiate_ids(b, tmp_path)
        assert len(ids) == len(set(ids)), f"duplicate ids in {b['id']}: {ids}"
        # one id per grid point (here: seeds x any ablation axis).
        n_points = 1
        for vals in b["grid"].values():
            n_points *= len(vals)
        assert len(ids) == n_points


# ---------------------------------------------------------------------------
# F2 — baseline head-to-head: percentile-norm vs default activation-scale.
# ---------------------------------------------------------------------------
def test_f2_batches_carry_both_baseline_arms():
    batches = em.gen_f2_batches(F_CELLS, seeds=(0, 1, 2))
    assert batches
    quant_key = "deployment_parameters.activation_scale_quantile"
    for b in batches:
        _assert_schema(b)
        assert quant_key in b["grid"], f"F2 {b['id']} missing the baseline axis"
        arms = b["grid"][quant_key]
        # The two arms: default activation-scale (0.99) AND percentile-norm (1.0).
        assert em.DEFAULT_ACTIVATION_SCALE_QUANTILE in arms
        assert em.PERCENTILE_NORM_QUANTILE in arms
        assert b.get("tags", {}).get("study") == "F2"


# ---------------------------------------------------------------------------
# F3 — dual-regime: from_scratch vs pretrained per (model, dataset).
# ---------------------------------------------------------------------------
def test_f3_batches_carry_both_regimes():
    batches = em.gen_f3_batches(F_CELLS, seeds=(0, 1))
    assert batches
    key = em.PRELOAD_KEY
    for b in batches:
        _assert_schema(b)
        assert key in b["grid"], f"F3 {b['id']} missing the regime axis"
        assert set(b["grid"][key]) == {False, True}
        assert b.get("tags", {}).get("study") == "F3"


# ---------------------------------------------------------------------------
# Aggregator: mean +/- CI on a synthetic ledger.
# ---------------------------------------------------------------------------
def _t_ci95_halfwidth(values):
    """Independent reference: Student-t 95% half-width about the mean."""
    import statistics as st

    n = len(values)
    if n < 2:
        return 0.0
    sd = st.stdev(values)
    se = sd / math.sqrt(n)
    return em.T_CRIT_95[n - 1] * se


SYN_F1 = [
    # cell A: deep_cnn / mnist / cascaded / d4  (3 seeds)
    {"study": "F1", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "seed": 0, "deployed_acc": 0.90, "run_id": "a0"},
    {"study": "F1", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "seed": 1, "deployed_acc": 0.92, "run_id": "a1"},
    {"study": "F1", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "seed": 2, "deployed_acc": 0.94, "run_id": "a2"},
    # cell B: lenet5 / fmnist / synchronized / d3 (2 seeds)
    {"study": "F1", "model": "lenet5", "dataset": "fmnist", "schedule": "synchronized",
     "depth": 3, "seed": 0, "deployed_acc": 0.80, "run_id": "b0"},
    {"study": "F1", "model": "lenet5", "dataset": "fmnist", "schedule": "synchronized",
     "depth": 3, "seed": 1, "deployed_acc": 0.84, "run_id": "b1"},
    # a non-F1 row that MUST be ignored by the F1 aggregator
    {"study": "F2", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "seed": 0, "deployed_acc": 0.10, "run_id": "junk"},
    # an F1 row with no metric -> dropped (cannot enter the mean)
    {"study": "F1", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "seed": 9, "deployed_acc": None, "run_id": "a_nan"},
]


def test_aggregate_f1_mean_and_ci():
    table = em.aggregate_f1(SYN_F1)
    assert len(table) == 2  # two cells
    by_cell = {tuple(r["cell"]): r for r in table}
    a = by_cell[("deep_cnn", "mnist", "cascaded", 4)]
    assert a["n_seeds"] == 3
    assert a["deployed_acc_mean"] == pytest.approx(0.92, abs=1e-12)
    assert a["ci95"] == pytest.approx(_t_ci95_halfwidth([0.90, 0.92, 0.94]), rel=1e-9)
    assert sorted(a["run_ids"]) == ["a0", "a1", "a2"]
    b = by_cell[("lenet5", "fmnist", "synchronized", 3)]
    assert b["n_seeds"] == 2
    assert b["deployed_acc_mean"] == pytest.approx(0.82, abs=1e-12)
    assert b["ci95"] == pytest.approx(_t_ci95_halfwidth([0.80, 0.84]), rel=1e-9)


def test_aggregate_f1_single_seed_zero_ci():
    rows = [{"study": "F1", "model": "m", "dataset": "d", "schedule": "s",
             "depth": 1, "seed": 0, "deployed_acc": 0.5, "run_id": "x"}]
    table = em.aggregate_f1(rows)
    assert len(table) == 1
    assert table[0]["n_seeds"] == 1
    assert table[0]["ci95"] == 0.0


# ---------------------------------------------------------------------------
# Aggregator: F2 head-to-head delta (percentile-norm - default).
# ---------------------------------------------------------------------------
SYN_F2 = [
    # cell A, default arm (0.99)
    {"study": "F2", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "activation_scale_quantile": 0.99, "seed": 0, "deployed_acc": 0.90,
     "run_id": "d0"},
    {"study": "F2", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "activation_scale_quantile": 0.99, "seed": 1, "deployed_acc": 0.92,
     "run_id": "d1"},
    # cell A, percentile-norm arm (1.0)
    {"study": "F2", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "activation_scale_quantile": 1.0, "seed": 0, "deployed_acc": 0.93,
     "run_id": "p0"},
    {"study": "F2", "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
     "depth": 4, "activation_scale_quantile": 1.0, "seed": 1, "deployed_acc": 0.95,
     "run_id": "p1"},
]


def test_aggregate_f2_head_to_head_delta():
    table = em.aggregate_f2(SYN_F2)
    assert len(table) == 1
    row = table[0]
    assert tuple(row["cell"]) == ("deep_cnn", "mnist", "cascaded", 4)
    assert row["default_mean"] == pytest.approx(0.91, abs=1e-12)
    assert row["percentile_mean"] == pytest.approx(0.94, abs=1e-12)
    # delta = percentile-norm - default = +0.03 -> +3.0 pp
    assert row["delta_pp"] == pytest.approx(3.0, abs=1e-9)
    assert row["n_default"] == 2 and row["n_percentile"] == 2


def test_aggregate_f2_skips_incomplete_pairs():
    rows = [r for r in SYN_F2 if r["activation_scale_quantile"] == 0.99]
    assert em.aggregate_f2(rows) == []  # no percentile arm -> no head-to-head


# ---------------------------------------------------------------------------
# Aggregator: F3 dual-regime delta (pretrained - from_scratch).
# ---------------------------------------------------------------------------
SYN_F3 = [
    {"study": "F3", "model": "vgg16", "dataset": "cifar10", "preload_weights": False,
     "seed": 0, "deployed_acc": 0.70, "run_id": "fs0"},
    {"study": "F3", "model": "vgg16", "dataset": "cifar10", "preload_weights": False,
     "seed": 1, "deployed_acc": 0.72, "run_id": "fs1"},
    {"study": "F3", "model": "vgg16", "dataset": "cifar10", "preload_weights": True,
     "seed": 0, "deployed_acc": 0.85, "run_id": "pt0"},
    {"study": "F3", "model": "vgg16", "dataset": "cifar10", "preload_weights": True,
     "seed": 1, "deployed_acc": 0.87, "run_id": "pt1"},
]


def test_aggregate_f3_dual_regime_delta():
    table = em.aggregate_f3(SYN_F3)
    assert len(table) == 1
    row = table[0]
    assert tuple(row["cell"]) == ("vgg16", "cifar10")
    assert row["from_scratch_mean"] == pytest.approx(0.71, abs=1e-12)
    assert row["pretrained_mean"] == pytest.approx(0.86, abs=1e-12)
    assert row["delta_pp"] == pytest.approx(15.0, abs=1e-9)
    assert row["n_from_scratch"] == 2 and row["n_pretrained"] == 2


# ---------------------------------------------------------------------------
# Markdown writers: produce a file with the table contents.
# ---------------------------------------------------------------------------
def test_write_findings_markdown(tmp_path):
    f1 = em.aggregate_f1(SYN_F1)
    f2 = em.aggregate_f2(SYN_F2)
    f3 = em.aggregate_f3(SYN_F3)
    out = tmp_path / "F_matrix.md"
    path = em.write_findings_markdown(str(out), f1=f1, f2=f2, f3=f3)
    assert os.path.isfile(path)
    text = open(path).read()
    assert "F1" in text and "F2" in text and "F3" in text
    # mean +/- CI rendered; head-to-head delta rendered; regime delta rendered.
    assert "0.9200" in text  # F1 cell-A mean
    assert "+3.00" in text   # F2 delta pp
    assert "+15.00" in text  # F3 delta pp


def test_write_findings_markdown_default_destination(tmp_path, monkeypatch):
    """With no explicit path the writer lands under docs/research/findings/."""
    monkeypatch.setattr(em, "FINDINGS_DIR", str(tmp_path / "findings"))
    path = em.write_findings_markdown(None, f1=em.aggregate_f1(SYN_F1))
    assert os.path.isfile(path)
    assert os.path.dirname(path) == str(tmp_path / "findings")


# ---------------------------------------------------------------------------
# read_ledger: tolerant JSONL reader (skips blank/garbage lines).
# ---------------------------------------------------------------------------
def test_read_ledger_skips_garbage(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        '{"study": "F1", "deployed_acc": 0.5}\n'
        "\n"
        "not json at all\n"
        '{"study": "F2", "deployed_acc": 0.6}\n'
    )
    rows = em.read_ledger(str(p))
    assert len(rows) == 2
    assert rows[0]["study"] == "F1" and rows[1]["study"] == "F2"
    assert em.read_ledger(str(tmp_path / "nope.jsonl")) == []


# ---------------------------------------------------------------------------
# Helper: drive scheduler.instantiate with a stubbed template loader.
# ---------------------------------------------------------------------------
_STUB_TEMPLATE = {
    "deployment_parameters": {"model_type": "deep_cnn", "model_config": {}},
    "data_provider_name": "MNIST_DataProvider",
    "pipeline_mode": "phased",
}


def _instantiate_ids(batch, tmp_path):
    import copy
    import json as _json

    tpl = tmp_path / "stub_template.json"
    tpl.write_text(_json.dumps(_STUB_TEMPLATE))
    b = copy.deepcopy(batch)
    b["template"] = os.path.relpath(str(tpl), sch.REPO)
    return [jid for jid, _ in sch.instantiate(b)]
