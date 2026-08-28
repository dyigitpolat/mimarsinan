"""NarrowConvNet: the vehicle's fan-in bound is ARCHITECTURAL, not a mapper result."""

import pytest
import torch

from mimarsinan.mapping.verification.streamed import streamed_span_report_model
from mimarsinan.models.vehicles.narrow_conv import NarrowConvNet
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def _fan_ins(model: NarrowConvNet) -> dict[str, int]:
    """Every post-stem layer's logical fan-in INCLUDING the bias slot."""
    fan_ins = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name != "stem.0":
            k_h, k_w = module.kernel_size
            fan_ins[name] = module.in_channels * k_h * k_w + 1
        elif isinstance(module, torch.nn.Linear):
            fan_ins[name] = module.in_features + 1
    return fan_ins


class TestNarrowConvNet:
    def test_forward_shape(self):
        model = NarrowConvNet((1, 28, 28), 10)
        assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)

    def test_the_collapse_conv_consumes_the_whole_residual_extent(self):
        model = NarrowConvNet((1, 28, 28), 10, stem_kernel=7, stem_stride=4,
                              body_blocks=2, body_channels=14, head_width=120)
        features = model.body(model.stem(torch.rand(1, 1, 28, 28)))
        assert features.shape[-2:] == model.collapse[0].kernel_size
        assert model.collapse(features).shape[-2:] == (1, 1)

    def test_post_stem_fan_in_is_bounded_by_the_declared_widths(self):
        model = NarrowConvNet((1, 28, 28), 10, stem_channels=14, stem_kernel=7,
                              stem_stride=4, body_blocks=2, body_channels=14,
                              head_width=120)
        fan_ins = _fan_ins(model)
        assert fan_ins["body.0"] == 9 * 14 + 1
        assert fan_ins["body.2"] == 9 * 14 + 1
        assert fan_ins["collapse.0"] == 2 * 2 * 14 + 1
        assert fan_ins["head"] == 120 + 1
        assert max(fan_ins.values()) <= 127

    def test_widening_the_head_does_not_widen_any_row(self):
        """head_width is the capacity knob that costs NEURONS, never axons."""
        narrow = _fan_ins(NarrowConvNet((1, 28, 28), 10, body_channels=14,
                                        head_width=32))
        wide = _fan_ins(NarrowConvNet((1, 28, 28), 10, body_channels=14,
                                      head_width=250))
        assert narrow["collapse.0"] == wide["collapse.0"]
        assert wide["head"] > narrow["head"]

    def test_deepening_the_body_does_not_widen_any_row(self):
        shallow = _fan_ins(NarrowConvNet((1, 32, 32), 10, body_blocks=1))
        deep = _fan_ins(NarrowConvNet((1, 32, 32), 10, body_blocks=3))
        assert max(deep.values()) <= max(shallow.values())

    def test_a_body_stage_that_cannot_reduce_further_is_refused(self):
        with pytest.raises(ValueError, match="cannot reduce further"):
            NarrowConvNet((1, 4, 4), 10, stem_stride=4, body_blocks=3)

    def test_a_non_chw_input_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            NarrowConvNet((28, 28), 10)

    def test_end_to_end_under_both_placements(self):
        model = NarrowConvNet((1, 28, 28), 10)
        model(torch.rand(1, 1, 28, 28))
        for placement in ("subsume", "offload"):
            report = streamed_span_report_model(
                model, (1, 28, 28), 10, encoding_placement=placement,
            )
            assert report.end_to_end, placement

    def test_registered_builder_builds(self):
        builder_cls = ModelRegistry.get_builder_cls("narrow_conv")
        builder = builder_cls("cpu", (1, 28, 28), 10, {})
        model = builder.build({})
        assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10)

    def test_the_builder_honors_its_configuration(self):
        builder_cls = ModelRegistry.get_builder_cls("narrow_conv")
        builder = builder_cls("cpu", (1, 28, 28), 10, {})
        model = builder.build({"stem_channels": 14, "stem_kernel": 7,
                               "stem_stride": 4, "body_blocks": 2,
                               "body_channels": 14, "head_width": 120})
        assert model.stem[0].out_channels == 14
        assert model.stem[0].stride == (4, 4)
        assert model.head.in_features == 120
        assert max(_fan_ins(model).values()) <= 127
