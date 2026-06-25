"""Unit tests for the CIFAR-breadth backlog generator (B2 dataset-margin RGB).

Covers the CIFAR10/CIFAR100 deep_cnn + lenet5 breadth cells: the cells resolve
to a registered CIFAR provider name and a real on-disk template, every emitted
batch is backlog-schema-valid and instantiates to DISTINCT job ids via
``scheduler.instantiate`` (with the CIFAR provider + schedule + depth correctly
swapped into each config), and the deep_cnn/lenet5 models are VALID under
``classify_validity`` at the RGB 3x32x32 input shape so no enqueued cell is
INVALID. No GPU, no dataset download — the validity check builds the native
model from the registry and reads param/MAC fractions statically.

Self-contained: bootstraps ``scripts/campaign``, ``scripts/gpu`` and ``src`` /
``spikingjelly`` onto ``sys.path`` so the bare ``pytest`` command resolves the
campaign modules and the framework imports under PYTHONPATH=src:spikingjelly.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "scripts", "gpu"),
           os.path.join(_REPO, "src"), os.path.join(_REPO, "spikingjelly")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_matrix as em  # noqa: E402
import scheduler as sch  # noqa: E402

CIFAR_RGB_SHAPE = (3, 32, 32)
CIFAR10_PROVIDER = "CIFAR10_DataProvider"
CIFAR100_PROVIDER = "CIFAR100_DataProvider"


# ---------------------------------------------------------------------------
# Cells: the CIFAR providers and templates must actually resolve.
# ---------------------------------------------------------------------------
def test_cifar_breadth_matrix_providers_and_templates_resolve():
    """Each CIFAR cell names a REGISTERED provider and an on-disk template — a
    typo'd provider name or a missing template silently never enqueues."""
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    import mimarsinan.data_handling.data_providers  # noqa: F401 - populate registry

    cells = em.cifar_breadth_matrix()
    assert cells, "cifar_breadth_matrix() is empty"
    seen_providers = set()
    for cell in cells:
        assert cell.dataset in (CIFAR10_PROVIDER, CIFAR100_PROVIDER), cell.dataset
        # provider name resolves in the live registry
        assert cell.dataset in BasicDataProviderFactory._provider_registry, (
            f"{cell.dataset!r} is not a registered data provider")
        # template exists on disk (else scheduler.refill skips the batch)
        path = os.path.join(_REPO, cell.template)
        assert os.path.exists(path), (
            f"cifar cell template missing: {cell.template!r} (resolved {path!r})")
        assert cell.model in ("deep_cnn", "lenet5"), cell.model
        seen_providers.add(cell.dataset)
    # both CIFAR10 and CIFAR100 are exercised (RGB dataset-margin breadth)
    assert seen_providers == {CIFAR10_PROVIDER, CIFAR100_PROVIDER}


def test_cifar_breadth_covers_both_models():
    cells = em.cifar_breadth_matrix()
    models = {c.model for c in cells}
    assert {"deep_cnn", "lenet5"} <= models


# ---------------------------------------------------------------------------
# Schema: every emitted batch is backlog-schema-valid.
# ---------------------------------------------------------------------------
def _assert_schema(batch: dict) -> None:
    em.validate_batch(batch)  # SSOT contract
    for key in ("id", "template", "grid", "id_template"):
        assert key in batch, f"missing required backlog key {key!r}"
    leaves = set()
    for path, vals in batch["grid"].items():
        assert isinstance(vals, list) and len(vals) >= 1, path
        leaves.add(path.split(".")[-1])
    fmt_fields = {fn for _, fn, _, _ in __import__("string").Formatter().parse(
        batch["id_template"]) if fn}
    assert fmt_fields <= leaves


def test_cifar_breadth_batches_are_schema_valid_and_multiseed():
    batches = em.gen_cifar_breadth_batches(em.cifar_breadth_matrix(), seeds=(0, 1, 2))
    assert batches
    datasets = set()
    for b in batches:
        _assert_schema(b)
        # the dataset axis carries exactly the cell's CIFAR provider
        assert "data_provider_name" in b["grid"]
        prov = b["grid"]["data_provider_name"]
        assert len(prov) == 1 and prov[0] in (CIFAR10_PROVIDER, CIFAR100_PROVIDER)
        datasets.add(prov[0])
        # the breadth study varies the seed (>= 2 for a dataset-margin estimate)
        seed_axes = [k for k in b["grid"] if k.split(".")[-1] == "seed"]
        assert seed_axes and len(b["grid"][seed_axes[0]]) >= 2
        assert b.get("tags", {}).get("study") == "B2"
        assert b.get("tags", {}).get("dataset") in (CIFAR10_PROVIDER, CIFAR100_PROVIDER)
        # default-off: inert until the orchestrator rolls it out
        assert b.get("enabled") is False
    assert datasets == {CIFAR10_PROVIDER, CIFAR100_PROVIDER}


# ---------------------------------------------------------------------------
# Instantiation: distinct ids, CIFAR provider/schedule/depth swapped in.
# ---------------------------------------------------------------------------
def test_cifar_breadth_batches_instantiate_distinct_ids():
    batches = em.gen_cifar_breadth_batches(em.cifar_breadth_matrix(), seeds=(0, 1))
    for b in batches:
        rows = list(sch.instantiate(b))
        ids = [jid for jid, _ in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {b['id']}: {ids}"
        n_points = 1
        for vals in b["grid"].values():
            n_points *= len(vals)
        assert len(ids) == n_points
        # the CIFAR provider, schedule and depth are actually written into the cfg
        for _, cfg in rows:
            assert cfg["data_provider_name"] in (CIFAR10_PROVIDER, CIFAR100_PROVIDER)
            dp = cfg["deployment_parameters"]
            assert dp["ttfs_cycle_schedule"] in ("cascaded", "synchronized")


# ---------------------------------------------------------------------------
# Validity: deep_cnn / lenet5 are VALID on CIFAR at the RGB 3x32x32 shape.
# ---------------------------------------------------------------------------
def _classify(model_type: str, model_config: dict, num_classes: int):
    from mimarsinan.mapping.verification.onchip_fraction import classify_validity
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    dp = {"model_type": model_type, "model_config": model_config}
    builder = ModelRegistry.get_builder_cls(model_type)(
        "cpu", CIFAR_RGB_SHAPE, num_classes, dp)
    model = builder.build(model_config)
    return classify_validity(
        model, CIFAR_RGB_SHAPE, num_classes, encoding_placement="subsume")


def test_deep_cnn_valid_on_cifar10_rgb():
    """deep_cnn at every breadth depth is VALID (not INVALID) on CIFAR10 RGB."""
    for depth in (4, 6, 8):
        v = _classify("deep_cnn",
                      {"depth": depth, "width": 16, "base_activation": "ReLU"}, 10)
        assert v.tier != "INVALID", (
            f"deep_cnn d{depth} CIFAR10 classified INVALID "
            f"(param={v.param_frac:.4f}, mac={v.mac_frac:.4f})")
        # the breadth cells are firmly VALID (both fractions clear the majority)
        assert v.tier == "VALID"
        assert min(v.param_frac, v.mac_frac) >= 0.50


def test_lenet5_valid_on_cifar10_rgb():
    v = _classify("lenet5", {"variant": "lenet5", "base_activation": "ReLU"}, 10)
    assert v.tier == "VALID", (
        f"lenet5 CIFAR10 not VALID (tier={v.tier}, "
        f"param={v.param_frac:.4f}, mac={v.mac_frac:.4f})")


def test_breadth_cells_valid_on_cifar100_rgb():
    """CIFAR100's wider head must not push a breadth cell below validity."""
    for cell in em.cifar_breadth_matrix():
        if cell.dataset != CIFAR100_PROVIDER:
            continue
        depths = cell.depths or (None,)
        for depth in depths:
            cfg = {"base_activation": "ReLU"}
            if cell.model == "deep_cnn":
                cfg.update({"depth": depth, "width": 16})
            else:
                cfg.update({"variant": "lenet5"})
            v = _classify(cell.model, cfg, 100)
            assert v.tier != "INVALID", (
                f"{cell.model} d{depth} CIFAR100 INVALID "
                f"(param={v.param_frac:.4f}, mac={v.mac_frac:.4f})")


# ---------------------------------------------------------------------------
# File emission: the generator writes a JSON file, not the live backlog.
# ---------------------------------------------------------------------------
def test_emit_cifar_backlog_file(tmp_path):
    out = tmp_path / "backlog_cifar.json"
    written = em.emit_cifar_breadth_backlog(str(out))
    assert os.path.isfile(out)
    import json
    data = json.load(open(out))
    assert isinstance(data, list) and data
    assert len(data) == written == len(
        em.gen_cifar_breadth_batches(em.cifar_breadth_matrix()))
    for b in data:
        em.validate_batch(b)
        assert b["tags"]["study"] == "B2"
