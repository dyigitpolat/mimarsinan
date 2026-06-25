"""F4 capstone: turn a trained ResNet-50 ANN into a MEASURED SNN deployment.

The ImageNet training (a separate Group-2 GPU run) writes a checkpoint
``runs/imagenet/resnet50.pt`` of the shape ``{"model": state_dict, "val_top1": float}``.
This harness:

  1. loads that checkpoint into a ``load_pretrained_resnet50(pretrained=False)`` trunk;
  2. runs it through the REAL SNN deploy path (``models.pretrained_bridge.deploy_and_eval``,
     LIF) on an ImageNet-val SUBSET sized for a feasible sim (``--num-eval``, default 256 --
     the LIF sim is slow, so a subset is the honest move and is recorded AS a subset);
  3. records the ANN top-1 (from the checkpoint), the deployed-SNN top-1, the validity
     tier (``classify_validity``), and a cost record (``extract_cost_record``);
  4. EMITS a campaign-shaped ledger row (model=resnet50, dataset=imagenet, regime,
     deployment_validity, deployed acc, ann acc, n_eval) appended to
     ``runs/campaign/ledger.jsonl`` via the campaign ``ledger-append`` convention.

It CONSUMES the bridge / provider / cost-extraction / ledger writer verbatim -- it
reimplements none of them. The REAL ImageNet deploy run is a SUPERVISED step (see the
module's ``main`` CLI); the tests exercise the wiring on a tiny CPU stand-in.

  CLI:
    PYTHONPATH=src:spikingjelly env/bin/python scripts/gpu/deploy_imagenet_snn.py \
        --checkpoint runs/imagenet/resnet50.pt --num-eval 256 --T 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from mimarsinan.chip_simulation.certification import CertificationCell  # noqa: E402
from mimarsinan.chip_simulation.cost_extraction import (  # noqa: E402
    CostRecord,
    extract_cost_record,
)
from mimarsinan.models.pretrained_bridge import (  # noqa: E402
    DeployedEval,
    deploy_and_eval,
    load_pretrained_resnet50,
)

IMAGENET_CLASSES = 1000
IMAGENET_SHAPE = (3, 224, 224)
DEPLOY_FIRING = "lif"
DEPLOY_BACKEND = "hcm"  # the SpikingHybridCoreFlow deploy executor (not SANA-FE)
DEFAULT_CHECKPOINT = os.path.join(_REPO, "runs", "imagenet", "resnet50.pt")
DEFAULT_LEDGER = os.path.join(_REPO, "runs", "campaign", "ledger.jsonl")
_LEDGER_APPEND_CLI = os.path.join(_REPO, "scripts", "campaign", "research_loop.py")


# ── checkpoint ──────────────────────────────────────────────────────────────────


def load_checkpoint_into_model(checkpoint_path: str, model: nn.Module) -> float:
    """Load a ``{"model": state_dict, "val_top1": float}`` checkpoint into ``model``.

    Returns the checkpoint's ANN ``val_top1`` (in PERCENT, as the trainer wrote it) or
    ``NaN`` when the checkpoint carries no ``val_top1`` (an honest 'unknown', not a
    crash). The state dict is loaded strictly so a shape/key mismatch is a precise,
    verbatim error rather than a silent partial load.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    val_top1 = ckpt.get("val_top1") if isinstance(ckpt, dict) else None
    return float(val_top1) if val_top1 is not None else float("nan")


# ── deploy + cost ────────────────────────────────────────────────────────────────


def deploy_eval(
    model: nn.Module,
    input_shape,
    num_classes: int,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    simulation_length: int = 4,
) -> DeployedEval:
    """Deploy ``model`` through the real LIF SNN path and report the deployed top-1.

    A thin pass-through to ``models.pretrained_bridge.deploy_and_eval`` (consumed
    verbatim): convert -> install LIF -> IR map -> hybrid HCM pack -> deployed
    ``SpikingHybridCoreFlow`` sim -> argmax top-1 over the eval subset.
    """
    return deploy_and_eval(
        model.eval(), tuple(input_shape), int(num_classes),
        eval_inputs, eval_targets,
        simulation_length=int(simulation_length), spiking_mode=DEPLOY_FIRING,
    )


def _deploy_sanafe_shaped_snapshot(deployed: DeployedEval) -> Dict[str, Any]:
    """A SANA-FE-snapshot-shaped projection of the DEPLOYED structure (cost-extract input).

    The LIF deploy path runs ``SpikingHybridCoreFlow``, NOT SANA-FE, so there is no
    energy / spike telemetry -- those stay 0.0 (honestly unmeasured on this path). The
    STRUCTURE the cost extractor keys on is real and measured: ``T`` (S_global), the
    neural segments (depth), one window of ``T`` timesteps per segment (latency), and
    the deployed hard-core count (area). One representative per-sample record.
    """
    timesteps = int(deployed.simulation_length)
    cores_per_segment = (
        deployed.hard_cores // deployed.neural_segments if deployed.neural_segments else 0
    )
    remainder = (
        deployed.hard_cores - cores_per_segment * deployed.neural_segments
        if deployed.neural_segments else 0
    )
    segments = []
    for i in range(deployed.neural_segments):
        n_cores = cores_per_segment + (1 if i < remainder else 0)
        segments.append({
            "timesteps_executed": timesteps,
            "per_core": [{"n_neurons": 0} for _ in range(n_cores)],
        })
    return {
        "aggregate": {
            "total_energy_mj": 0.0,
            "total_spikes": 0,
            "sample_count": int(deployed.num_samples),
        },
        "per_sample": [{"T": timesteps, "segments": segments}],
    }


def build_cost_record(
    deployed: DeployedEval, *, run_id: str, provenance: Optional[Dict[str, Any]] = None
) -> CostRecord:
    """Build a well-formed cost record for the deployed LIF run (consumes extract_cost_record).

    Keyed to the ``lif@hcm`` certification cell. ``acc_deploy`` is the deployed top-1;
    the structural cost axes (s_global / depth / cores / latency_steps) come from the
    deployed mapping. Energy / spikes are 0.0 (not measured on the HCM deploy path).
    """
    cell = CertificationCell(firing=DEPLOY_FIRING, sync=None, backend=DEPLOY_BACKEND)
    prov = {"run_id": run_id, "deploy_path": "SpikingHybridCoreFlow", "subset": True}
    if provenance:
        prov.update(provenance)
    return extract_cost_record(
        cell=cell,
        deployed_accuracy=float(deployed.accuracy),
        sanafe_snapshot=_deploy_sanafe_shaped_snapshot(deployed),
        provenance=prov,
    )


# ── ledger row (campaign shape) ──────────────────────────────────────────────────


def _acc_fraction_or_none(top1_percent: float) -> Optional[float]:
    """A percent top-1 -> a [0,1] fraction; ``None`` for NaN (JSON-safe 'unknown')."""
    if top1_percent is None or math.isnan(top1_percent):
        return None
    return float(top1_percent) / 100.0


def build_ledger_row(
    *,
    deployed: DeployedEval,
    ann_top1: float,
    validity_tier: str,
    cost_record: CostRecord,
    num_eval: int,
    run_id: str,
    is_subset: bool,
    model_name: str = "resnet50",
    dataset_name: str = "imagenet",
    regime: str = "pretrained",
) -> Dict[str, Any]:
    """A campaign-shaped ledger row for the deployed ImageNet SNN.

    The row carries the axes the coverage reader maps to a hypervolume cell (``model``
    vehicle, ``dataset``, ``spiking_mode`` firing, ``backend``, ``regime``) plus the
    verbatim ``deployment_validity`` tier and the MEASURED numbers of record: the
    deployed-SNN top-1, the ANN top-1 (as a [0,1] fraction), and the eval subset size.
    ``run_ids`` cites the run so the harvest de-dupes coverage per the campaign rule.
    """
    return {
        "model": str(model_name),
        "dataset": str(dataset_name),
        "regime": str(regime),
        "spiking_mode": DEPLOY_FIRING,
        "backend": DEPLOY_BACKEND,
        "deployment_validity": str(validity_tier),
        "deployed_acc": float(deployed.accuracy),
        "ann_acc": _acc_fraction_or_none(ann_top1),
        "ann_top1_percent": (None if ann_top1 is None or math.isnan(ann_top1) else float(ann_top1)),
        "n_eval": int(num_eval),
        "is_subset": bool(is_subset),
        "S": int(deployed.simulation_length),
        "depth": int(deployed.neural_segments),
        "hard_cores": int(deployed.hard_cores),
        "run_id": str(run_id),
        "run_ids": [str(run_id)],
        "cost_record": cost_record.to_dict(),
    }


def append_ledger_row(row: Dict[str, Any], *, ledger_path: str = DEFAULT_LEDGER) -> None:
    """Append one ledger row using the campaign ``ledger-append`` convention.

    Delegates to ``scripts/campaign/research_loop.py ledger-append`` (the SSOT writer:
    it stamps ``ts`` and writes one JSON line, making the dir). Falls back to the SAME
    one-line-JSON + default-``ts`` behavior in-process if that CLI cannot be spawned,
    so the wiring is testable without a subprocess but the byte shape is identical.
    """
    try:
        proc = subprocess.run(
            [sys.executable, _LEDGER_APPEND_CLI, "ledger-append", json.dumps(row)],
            cwd=_REPO,
            env={**os.environ, "MIM_CAMPAIGN_DIR": os.path.dirname(os.path.abspath(ledger_path))},
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    record = dict(row)
    record.setdefault("ts", time.time())
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── end-to-end harness ───────────────────────────────────────────────────────────


def run_deploy_capstone(
    *,
    checkpoint_path: str,
    model_factory: Callable[[], nn.Module],
    input_shape,
    num_classes: int,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    simulation_length: int = 4,
    ledger_path: str = DEFAULT_LEDGER,
    run_id: Optional[str] = None,
    is_subset: bool = True,
    model_name: str = "resnet50",
    dataset_name: str = "imagenet",
    regime: str = "pretrained",
) -> Dict[str, Any]:
    """The whole capstone: load -> deploy -> classify validity -> cost -> ledger row.

    ``model_factory`` builds a FRESH model instance each call (conversion mutates the
    model in place, so the deploy model and the validity-classification model must be
    distinct instances loaded from the SAME checkpoint). Returns the emitted ledger row
    (also appended to ``ledger_path``).
    """
    from mimarsinan.mapping.verification.onchip_fraction import classify_validity

    run_id = run_id or f"{model_name}-{dataset_name}-deploy-{int(time.time())}"

    deploy_model = model_factory()
    ann_top1 = load_checkpoint_into_model(checkpoint_path, deploy_model)
    deployed = deploy_eval(
        deploy_model, input_shape, num_classes,
        eval_inputs, eval_targets, simulation_length=simulation_length,
    )

    verdict_model = model_factory()
    load_checkpoint_into_model(checkpoint_path, verdict_model)
    verdict = classify_validity(verdict_model, tuple(input_shape), int(num_classes))

    cost_record = build_cost_record(deployed, run_id=run_id)
    row = build_ledger_row(
        deployed=deployed, ann_top1=ann_top1, validity_tier=verdict.tier,
        cost_record=cost_record, num_eval=int(eval_targets.shape[0]),
        run_id=run_id, is_subset=is_subset,
        model_name=model_name, dataset_name=dataset_name, regime=regime,
    )
    append_ledger_row(row, ledger_path=ledger_path)
    return row


# ── ImageNet-val subset (real run only; tests inject their own eval set) ──────────


def _imagenet_val_subset(num_eval: int, *, datasets_path: str, seed: int = 0):
    """Load a deterministic ImageNet-val SUBSET of ``num_eval`` (inputs, targets).

    Consumes the ``ImageNet_DataProvider`` (the project's provider) -- it reads the
    official ``split='val'`` set under ``IMAGENET_ROOT``. Real-run only (network /
    dataset present); the unit tests inject a tiny synthetic eval set instead.
    """
    from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
        ImageNet_DataProvider,
    )

    provider = ImageNet_DataProvider(datasets_path, seed=seed, batch_size=int(num_eval))
    loader = provider._get_test_dataset()  # the official held-out val split
    inputs, targets = [], []
    have = 0
    for i in range(len(loader)):
        x, y = loader[i]
        inputs.append(torch.as_tensor(x).unsqueeze(0))
        targets.append(int(y))
        have += 1
        if have >= num_eval:
            break
    return torch.cat(inputs, dim=0), torch.tensor(targets, dtype=torch.long)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="trained ResNet-50 checkpoint ({model state_dict, val_top1})")
    p.add_argument("--num-eval", type=int, default=256,
                   help="ImageNet-val SUBSET size (the LIF sim is slow; a subset is honest)")
    p.add_argument("--T", type=int, default=4, help="spiking window / temporal resolution")
    p.add_argument("--datasets-path", default=os.path.join(_REPO, "datasets"),
                   help="datasets dir (IMAGENET_ROOT is symlinked under it)")
    p.add_argument("--ledger", default=DEFAULT_LEDGER, help="ledger.jsonl to append to")
    p.add_argument("--run-id", default=None, help="run id cited in the ledger row")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not os.path.isfile(args.checkpoint):
        print(f"[deploy] checkpoint not found: {args.checkpoint} "
              f"(the ImageNet training writes it on completion)", file=sys.stderr)
        return 2

    print(f"[deploy] loading ImageNet-val subset (num_eval={args.num_eval}) ...", flush=True)
    eval_inputs, eval_targets = _imagenet_val_subset(
        args.num_eval, datasets_path=args.datasets_path, seed=args.seed
    )
    print(f"[deploy] deploying ResNet-50 as a LIF SNN (T={args.T}) on {args.num_eval} val images ...",
          flush=True)
    t0 = time.time()
    row = run_deploy_capstone(
        checkpoint_path=args.checkpoint,
        model_factory=lambda: load_pretrained_resnet50(IMAGENET_CLASSES, pretrained=False),
        input_shape=IMAGENET_SHAPE,
        num_classes=IMAGENET_CLASSES,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        simulation_length=args.T,
        ledger_path=args.ledger,
        run_id=args.run_id,
        is_subset=True,
        model_name="resnet50",
        dataset_name="imagenet",
        regime="pretrained",
    )
    wall_min = (time.time() - t0) / 60.0
    print(f"[deploy] DONE in {wall_min:.1f} min  ann_top1={row['ann_top1_percent']}  "
          f"deployed_top1={row['deployed_acc']:.4f}  tier={row['deployment_validity']}  "
          f"n_eval={row['n_eval']} (subset)  ledger+={args.ledger}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
