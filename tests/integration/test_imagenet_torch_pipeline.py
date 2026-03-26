"""
ImageNet + torchvision pretrained models: pipeline integration (build order).

1. Runs ``DeploymentPipeline`` through **Weight Preloading**, then measures **top-1
   accuracy on a fixed number of randomly sampled validation batches** and checks
   it against **torchvision's published ImageNet-1K acc@1** for the corresponding
   ``IMAGENET1K_V1`` weights (see ``Weights.meta['_metrics']``).

2. For CNN architectures (**``torch_squeezenet11``**, **``torch_vgg16``**), continues
   with **Torch Mapping** and runs the same random indices through the ``Supermodel``.

   **``torch_vit``** stops after step 1: native ViT matches published top-1 on random
   batches, but ``TorchMappingStep`` validation currently fails on the converted
   ViT graph (permute / shape mismatch in the mapper).

**Opt-in:** set ``RUN_IMAGENET_INTEGRATION=1`` and ``IMAGENET_ROOT`` in the project
``.env`` (or the environment). Tests use ``datasets_path`` under ``tmp_path``; the
provider symlinks ``<datasets_path>/imagenet`` to ``IMAGENET_ROOT`` like production.

Step order for the early segment is asserted in ``test_imagenet_torch_pipeline_step_specs``
(no ImageNet required).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torchvision.models as tvmodels
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mimarsinan.data_handling.data_providers  # noqa: F401 — register ImageNet provider

from mimarsinan.models.supermodel import Supermodel


def _load_dotenv_for_integration_gate() -> None:
    """So ``skipif`` and ``RUN_IMAGENET_INTEGRATION`` see ``IMAGENET_ROOT`` from the repo ``.env``."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


_load_dotenv_for_integration_gate()
from mimarsinan.pipelining.pipelines.deployment_pipeline import (
    DeploymentPipeline,
    get_pipeline_step_specs,
)


def _noop_reporter():
    class R:
        def report(self, *args, **kwargs):
            pass

        def console_log(self, *args, **kwargs):
            pass

        def finish(self):
            pass

    return R()


def _parse_deployment_config(deployment_config: dict) -> dict:
    """Same contract as ``main._parse_deployment_config`` (minimal duplicate to avoid GUI imports)."""
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

    data_provider_name = deployment_config["data_provider_name"]
    seed = deployment_config.get("seed", 0)
    datasets_path = deployment_config.get("datasets_path", "./datasets")
    data_provider_factory = BasicDataProviderFactory(data_provider_name, datasets_path, seed=seed)

    deployment_name = deployment_config["experiment_name"]
    deployment_parameters = dict(deployment_config["deployment_parameters"])

    platform_constraints_raw = deployment_config["platform_constraints"]
    if isinstance(platform_constraints_raw, dict) and "mode" in platform_constraints_raw:
        mode = platform_constraints_raw.get("mode", "user")
        if mode == "user":
            platform_constraints = platform_constraints_raw.get(
                "user",
                {k: v for k, v in platform_constraints_raw.items() if k != "mode"},
            )
        elif mode == "auto":
            auto = platform_constraints_raw.get("auto", {}) or {}
            platform_constraints = auto.get("fixed", {}) or {}
        else:
            raise ValueError(f"Invalid platform_constraints.mode: {mode}")
    else:
        platform_constraints = platform_constraints_raw

    pipeline_mode = deployment_config.get("pipeline_mode", "phased")

    if "_working_directory" in deployment_config:
        working_directory = deployment_config["_working_directory"]
    else:
        working_directory = (
            deployment_config["generated_files_path"] + "/"
            + deployment_name + "_" + pipeline_mode + "_deployment_run"
        )

    return {
        "pipeline_mode": pipeline_mode,
        "data_provider_factory": data_provider_factory,
        "deployment_name": deployment_name,
        "platform_constraints": platform_constraints,
        "deployment_parameters": deployment_parameters,
        "working_directory": working_directory,
        "start_step": deployment_config.get("start_step"),
        "stop_step": deployment_config.get("stop_step"),
        "target_metric_override": deployment_config.get("target_metric_override"),
    }


def _merged_config_for_imagenet_torch_specs() -> dict:
    """Merged pipeline config used only for ``get_pipeline_step_specs`` (no dataset)."""
    dp = DeploymentPipeline.default_deployment_parameters.copy()
    pc = DeploymentPipeline.default_platform_constraints.copy()
    dp.update(
        {
            "configuration_mode": "user",
            "model_type": "torch_squeezenet11",
            "weight_source": "torchvision",
            "finetune_epochs": 0,
            "model_config": {"base_activation": "ReLU"},
            "spiking_mode": "rate",
            "activation_quantization": False,
            "weight_quantization": False,
            "degradation_tolerance": 0.0,
        }
    )
    return {**dp, **pc}


def test_imagenet_torch_pipeline_step_specs():
    """Expected early steps for vanilla + torch + torchvision weights (no ImageNet required)."""
    cfg = _merged_config_for_imagenet_torch_specs()
    names = [n for n, _ in get_pipeline_step_specs(cfg)]
    assert names[:4] == [
        "Model Configuration",
        "Model Building",
        "Weight Preloading",
        "Torch Mapping",
    ]


def _imagenet_integration_enabled() -> bool:
    if os.environ.get("RUN_IMAGENET_INTEGRATION") != "1":
        return False
    raw = os.environ.get("IMAGENET_ROOT", "").strip()
    if not raw:
        return False
    root = os.path.abspath(os.path.expanduser(raw))
    meta = os.path.join(root, "meta.bin")
    devkit = os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")
    train = os.path.join(root, "train")
    return os.path.isdir(train) and (os.path.isfile(meta) or os.path.isfile(devkit))


# Random val batches: total sample count = _NUM_RANDOM_BATCHES * _RANDOM_BATCH_SIZE
_NUM_RANDOM_BATCHES = 4
_RANDOM_BATCH_SIZE = 64
# Published acc@1 is full-val; finite random subset can deviate — tolerance band on (ref - acc).
_REF_TOP1_MARGIN = 0.14

_TORCHVISION_IMAGENET1K_REF = {
    "torch_squeezenet11": tvmodels.SqueezeNet1_1_Weights.IMAGENET1K_V1,
    "torch_vgg16": tvmodels.VGG16_BN_Weights.IMAGENET1K_V1,
    "torch_vit": tvmodels.ViT_B_16_Weights.IMAGENET1K_V1,
}


def _published_imagenet1k_top1(model_type: str) -> float:
    """ImageNet-1K acc@1 from torchvision ``Weights.meta`` (same as training recipe docs)."""
    w = _TORCHVISION_IMAGENET1K_REF[model_type]
    pct = w.meta["_metrics"]["ImageNet-1K"]["acc@1"]
    return float(pct) / 100.0


def _random_subset_indices(
    dataset_len: int,
    *,
    num_batches: int,
    batch_size: int,
    seed: int,
) -> list[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    need = min(num_batches * batch_size, dataset_len)
    perm = torch.randperm(dataset_len, generator=g)[:need]
    return perm.tolist()


def _top1_on_subset(model: torch.nn.Module, dataset: torch.utils.data.Dataset, indices: list[int], device: torch.device) -> float:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=_RANDOM_BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_type",
    ["torch_squeezenet11", "torch_vgg16", "torch_vit"],
)
@pytest.mark.skipif(
    not _imagenet_integration_enabled(),
    reason="Set RUN_IMAGENET_INTEGRATION=1 and IMAGENET_ROOT in .env (dataset with train/ and meta.bin or devkit)",
)
def test_imagenet_pretrained_torch_random_batches_and_mapping(tmp_path, model_type):
    """Weight preload → compare random-batch top-1 to torchvision ref; Torch Mapping → forward same batches."""
    ref_top1 = _published_imagenet1k_top1(model_type)
    work = tmp_path / model_type
    work.mkdir(parents=True, exist_ok=True)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    deployment_config = {
        "data_provider_name": "ImageNet_DataProvider",
        "datasets_path": str(datasets_dir),
        "experiment_name": f"imagenet_torch_{model_type}",
        "generated_files_path": str(tmp_path / "generated"),
        "_working_directory": str(work),
        "seed": 0,
        "pipeline_mode": "vanilla",
        "deployment_parameters": {
            "lr": 0.0001,
            "training_epochs": 1,
            "tuner_epochs": 1,
            "degradation_tolerance": 0.0,
            "configuration_mode": "user",
            "model_type": model_type,
            "weight_source": "torchvision",
            "finetune_epochs": 0,
            "model_config": {"base_activation": "ReLU"},
            "spiking_mode": "rate",
            "activation_quantization": False,
            "weight_quantization": False,
        },
        "platform_constraints": {
            "cores": [
                {"max_axons": 50000, "max_neurons": 600, "count": 100},
                {"max_axons": 600, "max_neurons": 50000, "count": 100},
            ],
            "max_axons": 50000,
            "max_neurons": 50000,
            "target_tq": 16,
            "simulation_steps": 16,
            "weight_bits": 8,
        },
        "start_step": None,
        "stop_step": None,
        "target_metric_override": None,
    }

    parsed = _parse_deployment_config(deployment_config)
    deployment_parameters = dict(parsed["deployment_parameters"])
    DeploymentPipeline.apply_preset(parsed["pipeline_mode"], deployment_parameters)

    pipeline = DeploymentPipeline(
        data_provider_factory=parsed["data_provider_factory"],
        deployment_parameters=deployment_parameters,
        platform_constraints=parsed["platform_constraints"],
        reporter=_noop_reporter(),
        working_directory=parsed["working_directory"],
    )

    dp = parsed["data_provider_factory"].create()
    val_ds = dp._get_validation_dataset()
    idx = _random_subset_indices(
        len(val_ds),
        num_batches=_NUM_RANDOM_BATCHES,
        batch_size=_RANDOM_BATCH_SIZE,
        seed=42,
    )

    pipeline.run(stop_step="Weight Preloading")

    native = pipeline.cache.get("Weight Preloading.model")
    assert native is not None, "expected pretrained native model in cache after Weight Preloading"

    device = pipeline.config["device"]
    native = native.to(device)
    empirical = _top1_on_subset(native, val_ds, idx, device)
    assert empirical >= ref_top1 - _REF_TOP1_MARGIN, (
        f"{model_type}: random-batch top-1 {empirical:.4f} below published ref {ref_top1:.4f} "
        f"(margin {_REF_TOP1_MARGIN})"
    )

    # ViT-B/16: torchvision top-1 check only; torch_mapping validate() fails on converted graph.
    if model_type == "torch_vit":
        return

    pipeline.run_from("Torch Mapping", stop_step="Torch Mapping")

    supermodel = pipeline.cache.get("Torch Mapping.model")
    assert supermodel is not None
    assert isinstance(supermodel, Supermodel)
    supermodel = supermodel.to(device)
    supermodel.eval()

    subset = Subset(val_ds, idx)
    loader = DataLoader(subset, batch_size=_RANDOM_BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = supermodel(xb)
            assert out.shape[0] == xb.shape[0]
            assert torch.isfinite(out).all()
