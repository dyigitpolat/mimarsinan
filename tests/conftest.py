"""
Shared fixtures for mimarsinan test suite.

Provides:
- MockPipeline: lightweight pipeline stand-in for testing steps in isolation.
- Minimal model, config, IR graph, and platform constraint fixtures.
"""

import sys
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.layers import (
    TransformedActivation,
    ClampDecorator,
    QuantizeDecorator,
    LeakyGradReLU,
)
from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    PerceptronMapper,
    Ensure2DMapper,
    EinopsRearrangeMapper,
    ModuleMapper,
    ModelRepresentation,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode


# ---------------------------------------------------------------------------
# MockPipeline — lightweight stand-in for Pipeline
# ---------------------------------------------------------------------------

class _NoopReporter:
    """Absorbs all report calls silently."""
    def report(self, *args, **kwargs):
        pass
    def console_log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


class MockPipeline:
    """
    Minimal pipeline mock that satisfies PipelineStep's interface.

    Usage in a test::

        mock = MockPipeline(config={...})
        mock.seed("model", some_model)
        step = SomeStep(pipeline=mock)
        step.run()
        result = mock.cache["SomeStep.model"]
    """

    def __init__(self, config=None, working_directory=None):
        self.cache = {}
        self.key_translations = {}
        self.config = config or default_config()
        self.working_directory = working_directory or tempfile.mkdtemp()
        self.data_provider_factory = MockDataProviderFactory()
        self.loss = nn.CrossEntropyLoss()
        self.reporter = _NoopReporter()
        self._target_metric = 0.0

    # -- Pipeline interface used by PipelineStep --

    def get_entry(self, step, key):
        real_key = self._translate_key(step.name, key)
        return self.cache.get(real_key)

    def add_entry(self, step, key, obj, load_store_strategy="basic"):
        real_key = step.name + "." + key
        self.cache[real_key] = obj

    def update_entry(self, step, key, obj, load_store_strategy="basic"):
        old_key = self._translate_key(step.name, key)
        if old_key in self.cache:
            del self.cache[old_key]
        new_key = step.name + "." + key
        self.cache[new_key] = obj

    def get_target_metric(self):
        return self._target_metric

    def set_target_metric(self, v):
        self._target_metric = v

    # -- Helpers for tests --

    def seed(self, virtual_key, obj, *, step_name="Seed"):
        """Pre-populate a cache entry under a given producer step name."""
        real_key = step_name + "." + virtual_key
        self.cache[real_key] = obj
        return real_key

    def prepare_step(self, step):
        """Wire key translations so the step can read its requires/updates."""
        self.key_translations[step.name] = {}
        for req in step.requires:
            for k in self.cache:
                if k.endswith("." + req):
                    self.key_translations[step.name][req] = k
                    break
        for upd in step.updates:
            for k in self.cache:
                if k.endswith("." + upd):
                    self.key_translations[step.name][upd] = k
                    break

    def _translate_key(self, step_name, key):
        return self.key_translations.get(step_name, {}).get(key, step_name + "." + key)


# ---------------------------------------------------------------------------
# Minimal data provider for testing
# ---------------------------------------------------------------------------

class TinyDataset(torch.utils.data.Dataset):
    """10-sample random dataset for testing."""

    def __init__(self, input_shape=(1, 8, 8), num_classes=4, size=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TinyDataProvider(DataProvider):
    """Minimal DataProvider backed by TinyDataset."""

    def __init__(self, datasets_path="", *, seed=0,
                 input_shape=(1, 8, 8), num_classes=4, size=10):
        super().__init__(datasets_path, seed=seed)
        self._ds = TinyDataset(input_shape, num_classes, size)
        self._num_classes = num_classes

    def _get_training_dataset(self):
        return self._ds

    def _get_validation_dataset(self):
        return self._ds

    def _get_test_dataset(self):
        return self._ds

    def get_prediction_mode(self):
        return ClassificationMode(self._num_classes)

    def get_input_shape(self):
        return self._ds.data.shape[1:]

    def get_output_shape(self):
        return self._num_classes

    def get_training_batch_size(self):
        return min(4, len(self._ds))

    def get_validation_batch_size(self):
        return len(self._ds)

    def get_test_batch_size(self):
        return len(self._ds)


class MockDataProviderFactory:
    """Factory that returns TinyDataProvider."""

    def __init__(self, input_shape=(1, 8, 8), num_classes=4):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._provider = None

    def create(self):
        if self._provider is None:
            self._provider = TinyDataProvider(
                input_shape=self._input_shape,
                num_classes=self._num_classes,
            )
        return self._provider


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def default_config():
    return {
        "device": "cpu",
        "target_tq": 4,
        "weight_bits": 8,
        "training_epochs": 1,
        "lr": 0.001,
        "weight_quantization": False,
        "activation_quantization": False,
        "allow_axon_tiling": False,
        "spiking_mode": "rate",
        "firing_mode": "Default",
        "spike_generation_mode": "Deterministic",
        "max_simulation_samples": -1,
    }


# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------

class TinyPerceptronFlow(PerceptronFlow):
    """Two-perceptron MLP for testing (64 -> 16 -> 4)."""

    def __init__(self, input_shape=(1, 8, 8), num_classes=4):
        super().__init__("cpu")
        in_features = 1
        for d in input_shape:
            in_features *= d

        self.input_activation = nn.Identity()
        self.p1 = Perceptron(16, in_features, normalization=nn.BatchNorm1d(16))
        self.p2 = Perceptron(num_classes, 16)

        inp = InputMapper(input_shape)
        self._in_act_mapper = ModuleMapper(inp, self.input_activation)
        out = EinopsRearrangeMapper(self._in_act_mapper, "... c h w -> ... (c h w)")
        out = Ensure2DMapper(out)
        out = PerceptronMapper(out, self.p1)
        out = PerceptronMapper(out, self.p2)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        self._in_act_mapper.module = activation

    def forward(self, x):
        return self._mapper_repr(x)


def make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4, tq=4):
    """Build a minimal Supermodel for testing, with proper activation setup."""
    from mimarsinan.models.preprocessing.input_cq import InputCQ
    from mimarsinan.tuning.adaptation_manager import AdaptationManager

    flow = TinyPerceptronFlow(input_shape, num_classes)
    model = Supermodel("cpu", input_shape, num_classes, InputCQ(tq), flow, tq)

    am = AdaptationManager()
    cfg = default_config()
    for p in model.get_perceptrons():
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        am.update_activation(cfg, p)

    model.eval()
    with torch.no_grad():
        model(torch.randn(2, *input_shape))
    return model


def make_tiny_ir_graph(in_dim=8, hidden_dim=4, out_dim=4):
    """Build a minimal two-core IRGraph for testing."""
    w1 = np.random.randn(in_dim + 1, hidden_dim).astype(np.float32) * 0.1
    w2 = np.random.randn(hidden_dim + 1, out_dim).astype(np.float32) * 0.1

    src1 = np.array(
        [IRSource(node_id=-2, index=i) for i in range(in_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )
    core1 = NeuralCore(
        id=0, name="hidden",
        input_sources=src1,
        core_matrix=w1,
        threshold=1.0,
        latency=0,
    )

    src2 = np.array(
        [IRSource(node_id=0, index=i) for i in range(hidden_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )
    core2 = NeuralCore(
        id=1, name="output",
        input_sources=src2,
        core_matrix=w2,
        threshold=1.0,
        latency=1,
    )

    out_sources = np.array(
        [IRSource(node_id=1, index=i) for i in range(out_dim)],
        dtype=object,
    )
    return IRGraph(nodes=[core1, core2], output_sources=out_sources)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline(tmp_path):
    cfg = default_config()
    return MockPipeline(config=cfg, working_directory=str(tmp_path / "pipeline_cache"))

@pytest.fixture
def tiny_supermodel():
    return make_tiny_supermodel()

@pytest.fixture
def tiny_ir_graph():
    return make_tiny_ir_graph()

@pytest.fixture
def tiny_data_provider():
    return TinyDataProvider()

@pytest.fixture
def platform_constraints():
    return {
        "max_axons": 256,
        "max_neurons": 256,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
        "allow_axon_tiling": False,
        "weight_bits": 8,
        "target_tq": 4,
    }
