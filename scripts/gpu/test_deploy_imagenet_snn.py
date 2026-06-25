"""Unit tests for the ImageNet deploy capstone harness (F4).

Tests-first wiring on a TINY stand-in -- NO ImageNet, NO GPU, NO network. A small
pipeline-native conv classifier + a few synthetic 224x224-shaped images + a tiny
fake checkpoint stand in for the real ResNet-50 / ImageNet-val deploy. They assert
the harness:

  * loads a ``{"model": state_dict, "val_top1": float}`` checkpoint into a model;
  * calls ``deploy_and_eval`` and reads a genuine deployed top-1 off the real sim;
  * builds a well-formed cost record (consumes ``extract_cost_record``);
  * EMITS a ledger row of the campaign's shape (model/dataset/regime/
    deployment_validity/deployed acc/ann acc/n_eval) the coverage reader accepts.

The REAL ImageNet deploy run is a SUPERVISED post-build step (NOT run here).
"""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import deploy_imagenet_snn as dis  # noqa: E402

from mimarsinan.chip_simulation.coverage_ledger import (  # noqa: E402
    classify_validity_tier,
    row_to_cell,
)


# A small pipeline-native classifier (conv/bn/relu/pool/linear): the same op set as
# the ResNet bridge minus the residual add, so the deployed sim stays cheap. The
# hidden Linear is a genuine on-chip neural layer so it packs onto a real hard core.
class _TinyConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(4, 8)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.flatten(self.gap(x))
        return self.fc(self.relu2(self.hidden(x)))


_SHAPE = (3, 8, 8)   # tiny spatial -> few cores -> fast deployed sim
_T = 4               # tiny spiking window -> fast sim
_CLASSES = 4
_N = 6               # tiny eval subset
_ANN_TOP1 = 73.5     # stand-in checkpoint val_top1 (percent)


def _fake_checkpoint(tmp_path, model: nn.Module) -> str:
    path = os.path.join(str(tmp_path), "tiny.pt")
    torch.save({"model": model.state_dict(), "val_top1": _ANN_TOP1}, path)
    return path


def _fake_eval_set(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.rand(_N, *_SHAPE)
    y = torch.randint(0, _CLASSES, (_N,))
    return x, y


# ── checkpoint loading ────────────────────────────────────────────────────────


def test_load_checkpoint_into_model_recovers_weights_and_ann_top1(tmp_path):
    """load_checkpoint restores the state_dict bit-exactly and returns ann_top1."""
    src = _TinyConvNet(_CLASSES).eval()
    ckpt = _fake_checkpoint(tmp_path, src)

    dst = _TinyConvNet(_CLASSES).eval()
    # Perturb dst so a successful load is observable.
    with torch.no_grad():
        dst.fc.weight.add_(1.0)

    ann_top1 = dis.load_checkpoint_into_model(ckpt, dst)
    assert ann_top1 == pytest.approx(_ANN_TOP1)
    for (na, a), (nb, b) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert na == nb
        assert torch.equal(a, b), na


def test_load_checkpoint_missing_val_top1_is_nan(tmp_path):
    """A checkpoint without val_top1 yields NaN ann_top1 (honest 'unknown'), not a crash."""
    src = _TinyConvNet(_CLASSES).eval()
    path = os.path.join(str(tmp_path), "no_top1.pt")
    torch.save({"model": src.state_dict()}, path)
    dst = _TinyConvNet(_CLASSES).eval()
    ann_top1 = dis.load_checkpoint_into_model(path, dst)
    assert ann_top1 != ann_top1  # NaN


# ── deployed-eval + cost record ────────────────────────────────────────────────


def test_deploy_eval_returns_deployed_top1_from_real_sim():
    """deploy_eval converts->maps->packs->runs the real sim and reports a deployed top-1."""
    model = _TinyConvNet(_CLASSES).eval()
    x, y = _fake_eval_set()
    result = dis.deploy_eval(model, _SHAPE, _CLASSES, x, y, simulation_length=_T)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.num_samples == _N
    assert result.spiking_mode == "lif"
    assert result.neural_segments >= 1
    assert result.hard_cores >= 1
    # The deployed top-1 is exactly the argmax-vs-target rate of the sim logits.
    predicted = result.logits.argmax(dim=1)
    assert result.accuracy == pytest.approx(float((predicted == y).double().mean()), abs=1e-12)


def test_build_cost_record_is_well_formed_from_deployed_structure():
    """build_cost_record consumes extract_cost_record into a well-formed CostRecord."""
    model = _TinyConvNet(_CLASSES).eval()
    x, y = _fake_eval_set()
    result = dis.deploy_eval(model, _SHAPE, _CLASSES, x, y, simulation_length=_T)
    rec = dis.build_cost_record(result, run_id="tiny-deploy-0")
    # Keyed to the LIF deploy cell, carries the deployed structure as the cost tuple.
    assert rec.acc_deploy == pytest.approx(result.accuracy)
    assert rec.s_global == _T
    assert rec.depth == result.neural_segments
    assert rec.cores == result.hard_cores
    assert rec.latency_steps == result.neural_segments * _T
    assert rec.backend == dis.DEPLOY_BACKEND
    # Round-trips through the canonical cost-record dict form.
    from mimarsinan.chip_simulation.cost_extraction import CostRecord
    assert CostRecord.from_dict(rec.to_dict()) == rec


# ── ledger row schema ──────────────────────────────────────────────────────────


def test_build_ledger_row_has_campaign_schema_and_is_a_science_cell():
    """build_ledger_row emits a row the coverage reader maps to a real ImageNet cell."""
    model = _TinyConvNet(_CLASSES).eval()
    x, y = _fake_eval_set()
    result = dis.deploy_eval(model, _SHAPE, _CLASSES, x, y, simulation_length=_T)
    rec = dis.build_cost_record(result, run_id="tiny-deploy-0")
    row = dis.build_ledger_row(
        deployed=result,
        ann_top1=_ANN_TOP1,
        validity_tier="VALID",
        cost_record=rec,
        num_eval=_N,
        run_id="tiny-deploy-0",
        is_subset=True,
    )
    # Campaign row shape: the fields the coverage reader + harvest cite.
    assert row["model"] == "resnet50"
    assert row["dataset"] == "imagenet"
    assert row["regime"] == "pretrained"
    assert row["spiking_mode"] == "lif"
    assert row["deployment_validity"] == "VALID"
    assert row["deployed_acc"] == pytest.approx(result.accuracy)
    assert row["ann_acc"] == pytest.approx(_ANN_TOP1 / 100.0)
    assert row["n_eval"] == _N
    assert row["is_subset"] is True
    assert row["run_id"] == "tiny-deploy-0"
    assert "tiny-deploy-0" in row["run_ids"]
    # It carries the deployed cost tuple (so the row is self-contained for the scatter).
    assert row["cost_record"]["cores"] == result.hard_cores
    # The coverage reader treats it as a real science cell (valid tier) for the
    # resnet50/imagenet/lif/pretrained vehicle -- NOT a non-science run-status row.
    assert classify_validity_tier(row["deployment_validity"]) is not None
    cell = row_to_cell(row)
    assert cell is not None
    assert cell.vehicle == "resnet50"
    assert cell.dataset == "imagenet"
    assert cell.firing == "lif"
    assert cell.regime == "pretrained"


def test_ledger_row_flagged_and_invalid_tiers_round_trip():
    """The row schema preserves the verbatim validity tier (VALID_FLAGGED / INVALID)."""
    model = _TinyConvNet(_CLASSES).eval()
    x, y = _fake_eval_set()
    result = dis.deploy_eval(model, _SHAPE, _CLASSES, x, y, simulation_length=_T)
    rec = dis.build_cost_record(result, run_id="r")
    for tier, expected in (("VALID_FLAGGED", "VALID_FLAGGED"), ("INVALID", "INVALID")):
        row = dis.build_ledger_row(
            deployed=result, ann_top1=float("nan"), validity_tier=tier,
            cost_record=rec, num_eval=_N, run_id="r", is_subset=True,
        )
        assert row["deployment_validity"] == expected
        assert classify_validity_tier(row["deployment_validity"]).name == expected.replace("VALID_FLAGGED", "VALID_FLAGGED")
        # A NaN ann_acc is recorded as None (JSON-safe 'unknown'), not NaN.
        assert row["ann_acc"] is None


# ── ledger append (campaign convention) ─────────────────────────────────────────


def test_append_ledger_row_uses_campaign_convention(tmp_path):
    """append_ledger_row writes one JSON line per row + a default ts (campaign shape)."""
    ledger = os.path.join(str(tmp_path), "ledger.jsonl")
    row = {"model": "resnet50", "dataset": "imagenet", "deployment_validity": "VALID"}
    dis.append_ledger_row(row, ledger_path=ledger)
    dis.append_ledger_row(row, ledger_path=ledger)
    lines = [ln for ln in open(ledger).read().splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        parsed = json.loads(ln)
        assert parsed["model"] == "resnet50"
        assert "ts" in parsed  # the ledger-append convention stamps ts


# ── end-to-end harness wiring (tiny stand-in, no ImageNet) ──────────────────────


def test_run_deploy_capstone_end_to_end_on_tiny_standin(tmp_path):
    """The whole harness: checkpoint -> load -> deploy -> validity -> cost -> ledger row.

    Exercises run_deploy_capstone with a TINY model_factory + an injected fake eval
    set + a fake checkpoint, asserting it loads, deploys on the real sim, classifies
    validity, builds a cost record, and APPENDS exactly one well-formed ledger row.
    """
    src = _TinyConvNet(_CLASSES).eval()
    ckpt = _fake_checkpoint(tmp_path, src)
    ledger = os.path.join(str(tmp_path), "ledger.jsonl")
    x, y = _fake_eval_set(seed=3)

    row = dis.run_deploy_capstone(
        checkpoint_path=ckpt,
        model_factory=lambda: _TinyConvNet(_CLASSES),
        input_shape=_SHAPE,
        num_classes=_CLASSES,
        eval_inputs=x,
        eval_targets=y,
        simulation_length=_T,
        ledger_path=ledger,
        run_id="tiny-e2e-0",
        is_subset=True,
        model_name="resnet50",
        dataset_name="imagenet",
    )

    # A well-formed row was returned AND persisted.
    assert row["model"] == "resnet50"
    assert row["dataset"] == "imagenet"
    assert row["regime"] == "pretrained"
    assert row["n_eval"] == _N
    assert 0.0 <= row["deployed_acc"] <= 1.0
    assert row["ann_acc"] == pytest.approx(_ANN_TOP1 / 100.0)
    assert row["deployment_validity"] in ("VALID", "VALID_FLAGGED", "INVALID")
    assert classify_validity_tier(row["deployment_validity"]) is not None

    persisted = [json.loads(ln) for ln in open(ledger).read().splitlines() if ln.strip()]
    assert len(persisted) == 1
    assert persisted[0]["run_id"] == "tiny-e2e-0"
    assert "ts" in persisted[0]
    # The persisted row maps to the resnet50/imagenet science cell.
    cell = row_to_cell(persisted[0])
    assert cell is not None and cell.vehicle == "resnet50" and cell.dataset == "imagenet"
