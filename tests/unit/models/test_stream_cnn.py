"""[P4] StreamCNN: the streamed-lif conv vehicle is streamable by construction."""

import torch

from mimarsinan.mapping.verification.streamed import (
    assert_streamable_model_or_raise,
)
from mimarsinan.models.vehicles.stream_cnn import StreamCNN


class TestStreamCNN:
    def test_forward_shape(self):
        model = StreamCNN((1, 28, 28), 10)
        out = model(torch.rand(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_streamable_under_both_placements(self):
        model = StreamCNN((1, 28, 28), 10)
        model(torch.rand(1, 1, 28, 28))  # materialize
        for placement in ("subsume", "offload"):
            assert_streamable_model_or_raise(
                model, (1, 28, 28), 10, encoding_placement=placement,
            )

    def test_registered_builder_builds(self):
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        builder_cls = ModelRegistry.get_builder_cls("stream_cnn")
        builder = builder_cls("cpu", (1, 28, 28), 10, {})
        model = builder.build({})
        assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10)
