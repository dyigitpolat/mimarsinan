"""NF (PyTorch) vs SCM (hybrid) subsample parity under cycle-accurate LIF."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.nn.activations import LIFActivation, run_cycle_accurate
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.simulation_factory import run_trainer_metric
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from conftest import MockDataProviderFactory, default_config


class _PipelineStub:
    def __init__(self, config):
        self.config = config
        self.data_provider_factory = MockDataProviderFactory(
            input_shape=tuple(config["input_shape"]),
            num_classes=int(config["num_classes"]),
            size=32,
        )


def _lif_mlp_pipeline_config():
    cfg = default_config()
    cfg.update(
        {
            "spiking_mode": "lif",
            "cycle_accurate_lif_forward": True,
            "simulation_steps": 4,
            "max_simulation_samples": 24,
            "seed": 0,
            "spike_generation_mode": "Uniform",
            "input_shape": (8,),
            "num_classes": 3,
        }
    )
    return cfg


def _build_cycle_accurate_lif_model():
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    lif1 = LIFActivation(T=4, activation_scale=torch.tensor(1.0))
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    lif2 = LIFActivation(T=4, activation_scale=torch.tensor(1.0))
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)

    class _Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessor = torch.nn.Identity()
            self._repr = repr_
            # Register perceptrons as submodules so ``model.modules()`` reaches
            # their LIFActivations (needed by ``run_cycle_accurate``).
            self.p1 = p1
            self.p2 = p2

        def forward(self, x):
            return self._repr(x)

    model = _Flow()
    model.eval()
    return model, repr_


def test_nf_scm_subsample_parity_cycle_accurate() -> None:
    cfg = _lif_mlp_pipeline_config()
    pipeline = _PipelineStub(cfg)
    model, repr_ = _build_cycle_accurate_lif_model()
    T = int(cfg["simulation_steps"])

    class _CAWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.preprocessor = inner.preprocessor
            self._inner = inner

        def forward(self, x):
            return run_cycle_accurate(self._inner, x, T)

    nf_model = _CAWrapper(model)
    nf_acc = run_trainer_metric(pipeline, nf_model)

    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=32,
        max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    flow = SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid,
        simulation_length=T,
        preprocessor=model.preprocessor,
        firing_mode=cfg["firing_mode"],
        spike_mode=cfg["spike_generation_mode"],
        thresholding_mode=cfg["thresholding_mode"],
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )
    scm_acc = run_trainer_metric(pipeline, flow)

    assert abs(nf_acc - scm_acc) <= 1e-3, (
        f"NF subsample {nf_acc:.4f} vs SCM {scm_acc:.4f} (gap {abs(nf_acc - scm_acc):.4f})"
    )


def test_nf_scm_per_sample_output_parity_cycle_accurate() -> None:
    """Stronger: per-sample, per-output equality after Phase A boundary fix."""
    torch.manual_seed(0)
    cfg = _lif_mlp_pipeline_config()
    model, repr_ = _build_cycle_accurate_lif_model()
    T = int(cfg["simulation_steps"])

    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=32,
        max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    flow = SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid,
        simulation_length=T,
        preprocessor=model.preprocessor,
        firing_mode=cfg["firing_mode"],
        spike_mode=cfg["spike_generation_mode"],
        thresholding_mode=cfg["thresholding_mode"],
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )

    x = torch.rand(8, 8)
    with torch.no_grad():
        nf_out = run_cycle_accurate(model, x, T)
        scm_out = flow(x) / float(T)
    torch.testing.assert_close(nf_out, scm_out.to(torch.float32), atol=1e-6, rtol=0.0)
