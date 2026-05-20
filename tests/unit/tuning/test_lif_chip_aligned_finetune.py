"""``LifChipAlignedFinetuneTuner``: bounded KD finetune through the chip-aligned forward."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.tuning.tuners.lif_chip_aligned_finetune_tuner import (
    LifChipAlignedFinetuneTuner,
)


class _MockDataProvider:
    def __init__(self, in_dim=8, num_classes=3, n_samples=4):
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.n_samples = n_samples

    def get_input_shape(self):
        return (self.in_dim,)

    def get_prediction_mode(self):
        return type("PM", (), {"num_classes": self.num_classes})()

    def create_loss(self):
        return nn.CrossEntropyLoss()


class _MockDataProviderFactory:
    def __init__(self, **kw):
        self._kw = kw

    def create(self):
        return _MockDataProvider(**self._kw)


def _build_setup(T: int = 4):
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    lif1 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    lif2 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )

    class _Flow(nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessor = nn.Identity()
            self.p1 = p1
            self.p2 = p2
            self._repr = repr_

        def forward(self, x):
            return self._repr(x)

        def get_perceptrons(self):
            return [p1, p2]

    model = _Flow()

    class _DataLoader:
        def __iter__(self):
            torch.manual_seed(1)
            for _ in range(2):
                yield (torch.rand(4, 8), torch.randint(0, 3, (4,)))

    class _Factory:
        def create(self, **kwargs):
            return _DataLoader()

    class _Pipeline:
        def __init__(self):
            self.config = {
                "input_shape": (8,),
                "simulation_steps": T,
                "firing_mode": "Default",
                "spike_generation_mode": "Uniform",
                "thresholding_mode": "<=",
                "spiking_mode": "lif",
                "cycle_accurate_lif_forward": True,
                "device": "cpu",
                "lr": 1e-3,
                "batch_size": 4,
            }
            self.data_provider_factory = _MockDataProviderFactory(
                in_dim=8, num_classes=3, n_samples=4,
            )

    return model, hybrid, _Pipeline(), _Factory()


def test_finetune_updates_only_encoding_perceptron_weights(monkeypatch) -> None:
    """After finetune, encoding-layer weights change; non-encoding weights don't."""
    model, hybrid, pipeline, factory = _build_setup()

    # Snapshot all params before finetune.
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Monkey-patch the DataLoaderFactory used by the tuner to return our fixture loader.
    from mimarsinan.data_handling import data_loader_factory as dlf_mod
    monkeypatch.setattr(dlf_mod, "DataLoaderFactory", lambda _f: factory)

    tuner = LifChipAlignedFinetuneTuner(
        pipeline, model, hybrid,
        num_epochs=1, max_batches_per_epoch=2, lr=1e-3,
    )
    tuner.run()

    # p1 is the encoding layer — its weights should change.
    p1_w_change = (model.p1.layer.weight - before["p1.layer.weight"]).abs().sum().item()
    assert p1_w_change > 0, "encoding perceptron weights should be updated"

    # p2 is NOT marked is_encoding_layer — its weights should be frozen.
    p2_w_change = (model.p2.layer.weight - before["p2.layer.weight"]).abs().sum().item()
    assert p2_w_change == 0, (
        f"non-encoding perceptron weights should be frozen; changed by {p2_w_change}"
    )


def test_finetune_restores_model_forward_and_grad_state(monkeypatch) -> None:
    model, hybrid, pipeline, factory = _build_setup()
    from mimarsinan.data_handling import data_loader_factory as dlf_mod
    monkeypatch.setattr(dlf_mod, "DataLoaderFactory", lambda _f: factory)

    pre_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    assert "forward" not in model.__dict__

    tuner = LifChipAlignedFinetuneTuner(
        pipeline, model, hybrid,
        num_epochs=1, max_batches_per_epoch=1,
    )
    tuner.run()

    # Forward unpatched.
    assert "forward" not in model.__dict__
    # requires_grad restored.
    post_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    assert post_requires_grad == pre_requires_grad


def test_finetune_restores_state_on_exception(monkeypatch) -> None:
    """If training raises, the tuner must still uninstall the wrapper + restore grad."""
    model, hybrid, pipeline, factory = _build_setup()
    from mimarsinan.data_handling import data_loader_factory as dlf_mod
    monkeypatch.setattr(dlf_mod, "DataLoaderFactory", lambda _f: factory)

    pre_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}

    # Force the loss to raise inside the training loop.
    import mimarsinan.tuning.tuners.lif_chip_aligned_finetune_tuner as tuner_mod
    orig = tuner_mod._KDChipAlignedLoss

    class _RaisingLoss(orig):
        def __call__(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(tuner_mod, "_KDChipAlignedLoss", _RaisingLoss)

    tuner = LifChipAlignedFinetuneTuner(
        pipeline, model, hybrid,
        num_epochs=1, max_batches_per_epoch=1,
    )
    try:
        tuner.run()
    except RuntimeError:
        pass

    assert "forward" not in model.__dict__
    post_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    assert post_requires_grad == pre_requires_grad
