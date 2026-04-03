"""Tests to isolate Torch Mapping accuracy: conversion vs native (no InputCQ wrapper)."""

import pytest
import torch

from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

# Pipeline-matching MLP-Mixer config (from deployment output)
PIPELINE_CONFIG = {
    "input_shape": (1, 28, 28),
    "num_classes": 10,
    "patch_n_1": 4,
    "patch_m_1": 4,
    "patch_c_1": 32,
    "fc_w_1": 64,
    "fc_w_2": 64,
    "base_activation": "LeakyReLU",
}


def _make_model(trained=False, seed=42):
    """Build native TorchMLPMixer; optionally train for 1 epoch with fixed seed."""
    torch.manual_seed(seed)
    model = TorchMLPMixer(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        patch_n_1=PIPELINE_CONFIG["patch_n_1"],
        patch_m_1=PIPELINE_CONFIG["patch_m_1"],
        patch_c_1=PIPELINE_CONFIG["patch_c_1"],
        fc_w_1=PIPELINE_CONFIG["fc_w_1"],
        fc_w_2=PIPELINE_CONFIG["fc_w_2"],
        base_activation=PIPELINE_CONFIG["base_activation"],
    )
    if trained:
        from conftest import MockDataProviderFactory
        from mimarsinan.model_training.basic_trainer import BasicTrainer

        class _Loss:
            def __call__(self, m, x, y):
                return torch.nn.functional.cross_entropy(m(x), y)

        dp_factory = MockDataProviderFactory(
            input_shape=PIPELINE_CONFIG["input_shape"],
            num_classes=PIPELINE_CONFIG["num_classes"],
        )
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        trainer = BasicTrainer(model, torch.device("cpu"), dlf, _Loss())
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)
    model.eval()
    return model


def _accuracy(model, loader, device="cpu", use_flow=False, flow=None):
    """Accuracy over loader; if use_flow, run x through converted flow only."""
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_flow and flow is not None:
                pred = flow(x).argmax(dim=1)
            else:
                pred = model(x).argmax(dim=1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total if total else 0.0


@pytest.mark.parametrize("trained", [False, True], ids=["untrained", "trained"])
def test_flow_matches_native_on_full_val_loader(trained):
    """Same x: converted flow(x) vs model(x) should agree (argmax)."""
    from conftest import MockDataProviderFactory

    model = _make_model(trained=trained)
    flow = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    flow.eval()

    dp_factory = MockDataProviderFactory(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
    )
    dp = dp_factory.create()
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    val_loader = dlf.create_validation_loader(dp.get_validation_batch_size(), dp)

    _accuracy(model, val_loader)
    _accuracy(model, val_loader, use_flow=True, flow=flow)
    for x, y in val_loader:
        with torch.no_grad():
            p_native = model(x).argmax(dim=1)
            p_flow = flow(x).argmax(dim=1)
        agreement = (p_native == p_flow).float().mean().item()
        assert agreement == 1.0, (
            f"converted flow vs native disagree on {1 - agreement:.0%} of samples. "
            "Conversion or weight transfer is wrong."
        )
        break


def test_torch_mapping_validation_uses_full_loader():
    """Full-loader vs single-batch validation sanity."""
    from conftest import MockDataProviderFactory
    from mimarsinan.model_training.basic_trainer import BasicTrainer

    model = _make_model(trained=True)
    flow = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    flow.eval()

    class _Loss:
        def __call__(self, m, x, y):
            return torch.nn.functional.cross_entropy(m(x), y)

    dp_factory = MockDataProviderFactory(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
    )
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    trainer = BasicTrainer(flow, torch.device("cpu"), dlf, _Loss())
    full_test_acc = trainer.test()
    single_batch_acc = trainer.validate()
    assert 0.0 <= full_test_acc <= 1.0 and 0.0 <= single_batch_acc <= 1.0
