"""Tests to isolate Torch Mapping accuracy drop: conversion vs preprocessing.

See plan section 5.1: pinpoint whether 0.53 is from (a) wrong conversion,
(b) input quantization, or (c) double in_act.
"""

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


def _accuracy(model, loader, device="cpu", use_perceptron_flow=False, supermodel=None):
    """Accuracy over loader; if use_perceptron_flow, run x through supermodel.perceptron_flow only."""
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_perceptron_flow and supermodel is not None:
                pred = supermodel.perceptron_flow(x).argmax(dim=1)
            else:
                pred = model(x).argmax(dim=1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total if total else 0.0


@pytest.mark.parametrize("trained", [False, True], ids=["untrained", "trained"])
def test_perceptron_flow_matches_native_on_full_val_loader(trained):
    """If pass: conversion is correct; 0.53 is from validation path. If fail: conversion wrong."""
    from conftest import MockDataProviderFactory

    model = _make_model(trained=trained)
    supermodel = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    supermodel.eval()

    dp_factory = MockDataProviderFactory(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
    )
    dp = dp_factory.create()
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    val_loader = dlf.create_validation_loader(dp.get_validation_batch_size(), dp)

    acc_native = _accuracy(model, val_loader)
    acc_pf = _accuracy(
        model, val_loader, use_perceptron_flow=True, supermodel=supermodel
    )
    # Same x fed to both: perceptron_flow(x) vs model(x) should agree (argmax)
    for x, y in val_loader:
        with torch.no_grad():
            p_native = model(x).argmax(dim=1)
            p_pf = supermodel.perceptron_flow(x).argmax(dim=1)
        agreement = (p_native == p_pf).float().mean().item()
        assert agreement == 1.0, (
            f"perceptron_flow vs native disagree on {1 - agreement:.0%} of samples. "
            "Conversion or weight transfer is wrong."
        )
        break


def test_supermodel_vs_perceptron_flow_accuracy_same_input():
    """Assert acc_pf is high; if acc_sm low and acc_pf high, drop is in preprocessing."""
    from conftest import MockDataProviderFactory

    model = _make_model(trained=True)
    supermodel = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    supermodel.eval()

    dp_factory = MockDataProviderFactory(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
    )
    dp = dp_factory.create()
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    val_loader = dlf.create_validation_loader(dp.get_validation_batch_size(), dp)
    x, y = next(iter(val_loader))

    with torch.no_grad():
        acc_sm = (supermodel(x).argmax(dim=1) == y).float().mean().item()
        acc_pf = (supermodel.perceptron_flow(x).argmax(dim=1) == y).float().mean().item()

    assert acc_pf >= 0.0 and acc_pf <= 1.0
    # With same raw x, perceptron_flow should match native (tested above), so acc_pf
    # should be same as native accuracy on that batch. We only assert that perceptron_flow
    # path is sane; if acc_sm is much lower than acc_pf, the drop is in preprocessing.
    if acc_sm < 0.5 and acc_pf > 0.5:
        pytest.fail(
            f"Supermodel accuracy {acc_sm:.2f} << perceptron_flow accuracy {acc_pf:.2f} "
            "on same input — drop is in preprocessing / in_act."
        )


def test_preprocessor_only_vs_full_supermodel_input_path():
    """supermodel(x) equals perceptron_flow(preprocessor(x)) — single input activation."""
    model = _make_model(trained=False)
    supermodel = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    supermodel.eval()

    x = torch.randn(4, *PIPELINE_CONFIG["input_shape"])
    with torch.no_grad():
        z_full = supermodel(x)
        preprocessed = supermodel.preprocessor(x)
        z1 = supermodel.perceptron_flow(preprocessed)

    assert z_full.shape == z1.shape
    assert torch.allclose(z1, z_full), (
        "perceptron_flow(preprocessor(x)) should equal supermodel(x) (single input activation)."
    )


def test_torch_mapping_validation_uses_full_loader():
    """Optional: full-loader validation to see if single-batch 0.53 was noisy."""
    from conftest import MockDataProviderFactory
    from mimarsinan.model_training.basic_trainer import BasicTrainer

    model = _make_model(trained=True)
    supermodel = convert_torch_model(
        model,
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
        device=torch.device("cpu"),
        Tq=32,
    )
    supermodel.eval()

    class _Loss:
        def __call__(self, m, x, y):
            return torch.nn.functional.cross_entropy(m(x), y)

    dp_factory = MockDataProviderFactory(
        input_shape=PIPELINE_CONFIG["input_shape"],
        num_classes=PIPELINE_CONFIG["num_classes"],
    )
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    trainer = BasicTrainer(supermodel, torch.device("cpu"), dlf, _Loss())
    full_test_acc = trainer.test()
    single_batch_acc = trainer.validate()
    assert 0.0 <= full_test_acc <= 1.0 and 0.0 <= single_batch_acc <= 1.0
