"""Ratchet: ``encoding_layer_placement`` is honored for EVERY registered builder.

The defect this guards was model-type-specific and silent: ``simple_mlp`` (the
only ``native``-category model) baked ``is_encoding_layer=True`` in its builder,
and the placement-applying call site sat on the FX path that ``native`` models
never take — so flipping the knob changed nothing, for that one builder, with no
error anywhere. A per-model test would not have found it; this one asks the
generic question of every builder in the registry:

    flipping ``subsume -> offload`` must CHANGE the marked-encoder set,
    unless the model has no encoding layer at all (an empty subsume set).

The spec table must cover the registry exactly, so a newly registered builder
fails here until someone states how to build it — uncovered builders are how the
first no-op survived.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import resolved_encoding_placement

# model_type -> (input_shape, num_classes, model_config, pipeline_config_extra)
# The smallest build each builder accepts: the ratchet asks a structural
# question, so a 16-wide vehicle answers it as well as a deployed one.
BUILD_SPECS: dict[str, tuple] = {
    "simple_mlp": ((1, 8, 8), 4, {"mlp_width_1": 16, "mlp_width_2": 8}, {}),
    "deep_mlp": ((1, 8, 8), 4, {"depth": 3, "width": 16}, {}),
    "deep_cnn": ((1, 16, 16), 4, {"depth": 4, "width": 8}, {}),
    "lenet5": ((1, 28, 28), 10, {}, {}),
    "stream_cnn": ((1, 8, 8), 4, {}, {}),
    "narrow_conv": ((1, 8, 8), 4, {"body_blocks": 1}, {}),
    "mlp_mixer": (
        (1, 8, 8), 4,
        {"patch_n_1": 2, "patch_m_1": 2, "patch_c_1": 8, "fc_w_1": 16, "fc_w_2": 16},
        {},
    ),
    "mlp_mixer_core": (
        (1, 8, 8), 4,
        {"patch_n_1": 2, "patch_m_1": 2, "patch_c_1": 8, "fc_w_1": 16,
         "fc_w_2": 16, "num_blocks": 1},
        {},
    ),
    "torch_sequential_linear": ((16,), 4, {"hidden_dims": [16, 8]}, {}),
    "torch_sequential_conv": (
        (1, 16, 16), 4, {"conv_out_channels": 4, "hidden_dims": [16]}, {},
    ),
    # The factory takes the model_config alone (see TorchCustomBuilder.build).
    "torch_custom": (
        (1, 8, 8), 4, {},
        {"model_factory": lambda _config: torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )},
    ),
    "torch_vit_leaf": ((1, 32, 32), 10, {}, {}),
    "torch_squeezenet11": ((1, 28, 28), 10, {}, {}),
    "cifar_vgg8": ((3, 32, 32), 10, {}, {}),
    # Heavy fixed architectures: same question, minutes of tracing.
    "cifar_resnet20": ((3, 32, 32), 10, {}, {}),
    "cifar_vit_leaf": ((3, 32, 32), 10, {}, {}),
    "torch_vgg16": ((1, 28, 28), 10, {}, {}),
    "torch_vit": ((1, 28, 28), 10, {}, {}),
    "torch_resnet50": ((1, 28, 28), 10, {}, {}),
}

SLOW = frozenset({
    "cifar_resnet20", "cifar_vit_leaf", "torch_vgg16", "torch_vit", "torch_resnet50",
})

BUILDER_PARAMS = [
    pytest.param(mid, marks=pytest.mark.slow) if mid in SLOW else mid
    for mid in sorted(BUILD_SPECS)
]


def _build(model_type, *, placement):
    input_shape, num_classes, model_config, extra = BUILD_SPECS[model_type]
    builder = BUILDERS_REGISTRY[model_type](
        "cpu", input_shape, num_classes, {"target_tq": 32, "device": "cpu", **extra}
    )
    model = build_model(builder, model_config, encoding_placement=placement)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    return model, input_shape, num_classes


def _flow(model, input_shape, num_classes, placement):
    if hasattr(model, "get_mapper_repr"):
        return model
    return convert_torch_model(
        model, tuple(input_shape), int(num_classes),
        encoding_layer_placement=placement,
    )


def _marked_positions(flow):
    """Marked encoders by POSITION, so two independently built flows compare."""
    return {i for i, p in enumerate(flow.get_perceptrons()) if p.is_encoding_layer}


def _built_flow(model_type, placement):
    """The full build path this builder's model takes for ``placement``."""
    model, input_shape, num_classes = _build(model_type, placement=placement)
    return _flow(model, input_shape, num_classes, placement)


def test_spec_table_covers_the_registry_exactly():
    """A builder with no spec is a builder this ratchet never asks about."""
    registered = set(ModelRegistry.builder_classes())
    assert set(BUILD_SPECS) == registered, (
        "every registered builder must declare how it is built here, or the "
        "placement no-op can reappear unnoticed in the uncovered one: "
        f"missing={sorted(registered - set(BUILD_SPECS))} "
        f"stale={sorted(set(BUILD_SPECS) - registered)}"
    )


@pytest.mark.parametrize("model_type", BUILDER_PARAMS)
def test_builder_bakes_no_placement_decision(model_type):
    """A builder that returns a flow must return it UNRESOLVED.

    ``SimpleMLPBuilder`` failing exactly this assertion is the reported defect.
    """
    input_shape, num_classes, model_config, extra = BUILD_SPECS[model_type]
    builder = BUILDERS_REGISTRY[model_type](
        "cpu", input_shape, num_classes, {"target_tq": 32, "device": "cpu", **extra}
    )
    model = builder.build(model_config)
    if not hasattr(model, "get_mapper_repr"):
        pytest.skip(f"{model_type} builds a torch module; conversion resolves placement")

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    marked = [p for p in model.get_perceptrons() if p.is_encoding_layer]
    assert marked == [], (
        f"{model_type} baked an encoding-layer mark at build; the configured "
        "placement can then never change it (the simple_mlp no-op)"
    )
    assert resolved_encoding_placement(model.get_mapper_repr()) is None


@pytest.mark.parametrize("model_type", BUILDER_PARAMS)
def test_flipping_the_placement_changes_the_marked_encoders(model_type):
    """Two independent builds, one per placement — the knob's end-to-end claim."""
    subsumed_flow = _built_flow(model_type, "subsume")
    offloaded_flow = _built_flow(model_type, "offload")

    subsumed = _marked_positions(subsumed_flow)
    offloaded = _marked_positions(offloaded_flow)

    assert resolved_encoding_placement(subsumed_flow.get_mapper_repr()) == "subsume"
    assert resolved_encoding_placement(offloaded_flow.get_mapper_repr()) == "offload"
    assert offloaded == set(), (
        f"{model_type}: offload must leave NO host-side encoder mark "
        "(the encoding layer maps on chip)"
    )
    if subsumed:
        assert subsumed != offloaded, (
            f"{model_type}: flipping subsume->offload changed nothing — the "
            "placement knob is a no-op for this builder"
        )
    else:
        # The explicit declaration, read off the model: nothing in this graph
        # starts a neural segment, so there is no encoding layer to place.
        assert not _marked_positions(subsumed_flow)
