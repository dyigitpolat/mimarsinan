"""NarrowConvNet: the vehicle's fan-in bound is ARCHITECTURAL, not a mapper result."""

import pytest
import torch

from mimarsinan.mapping.verification.streamed import streamed_span_report_model
from mimarsinan.models.vehicles.narrow_conv import (
    ACTIVATED_READOUT,
    NarrowConvNet,
)
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

    def test_the_collapse_consumes_the_whole_residual_extent(self):
        """An event-serial fold reproduces a WHOLE-INPUT weight, never an unfold."""
        model = NarrowConvNet((1, 28, 28), 10, stem_kernel=7, stem_stride=4,
                              body_blocks=2, body_channels=14,
                              trunk_width=120).eval()
        features = model.body(model.stem(torch.rand(1, 1, 28, 28)))
        assert model.trunk[0].kernel_size == features.shape[-2:]
        assert model.trunk(features).shape[-2:] == (1, 1)

    def test_nothing_reshapes_between_the_hops(self):
        """A Flatten between two hops carries no event train; there is exactly one, last."""
        model = NarrowConvNet((1, 28, 28), 10, readout=ACTIVATED_READOUT)
        staged = [m for stage in (model.stem, model.body, model.trunk, model.head)
                  for m in stage]
        assert not any(isinstance(m, (torch.nn.Flatten, torch.nn.Linear))
                       for m in staged)
        assert isinstance(model.flatten, torch.nn.Flatten)

    def test_post_stem_fan_in_is_bounded_by_the_declared_widths(self):
        model = NarrowConvNet((1, 28, 28), 10, stem_channels=14, stem_kernel=7,
                              stem_stride=4, body_blocks=2, body_channels=14,
                              trunk_width=120, trunk_blocks=2)
        fan_ins = _fan_ins(model)
        assert fan_ins["body.0"] == 9 * 14 + 1
        assert fan_ins["body.3"] == 9 * 14 + 1
        assert fan_ins["trunk.0"] == 2 * 2 * 14 + 1
        assert fan_ins["trunk.3"] == 120 + 1
        assert fan_ins["head.0"] == 120 + 1
        assert max(fan_ins.values()) <= 127

    def test_a_bodyless_build_hands_the_stem_map_straight_to_the_collapse(self):
        model = NarrowConvNet((1, 28, 28), 10, stem_channels=14, stem_kernel=13,
                              stem_stride=10, body_blocks=0, trunk_width=120)
        assert len(model.body) == 0
        assert _fan_ins(model)["trunk.0"] == 3 * 3 * 14 + 1

    def test_widening_the_trunk_does_not_widen_the_collapse(self):
        """trunk_width is the capacity knob that costs NEURONS, never entry axons."""
        narrow = _fan_ins(NarrowConvNet((1, 28, 28), 10, body_channels=14,
                                        trunk_width=32))
        wide = _fan_ins(NarrowConvNet((1, 28, 28), 10, body_channels=14,
                                      trunk_width=250))
        assert narrow["trunk.0"] == wide["trunk.0"]
        assert wide["head.0"] > narrow["head.0"]

    def test_deepening_the_body_does_not_widen_any_row(self):
        shallow = _fan_ins(NarrowConvNet((1, 32, 32), 10, body_blocks=1))
        deep = _fan_ins(NarrowConvNet((1, 32, 32), 10, body_blocks=3))
        assert max(deep.values()) <= max(shallow.values())

    def test_deepening_the_trunk_does_not_widen_any_row(self):
        one = _fan_ins(NarrowConvNet((1, 28, 28), 10, trunk_blocks=1))
        three = _fan_ins(NarrowConvNet((1, 28, 28), 10, trunk_blocks=3))
        assert max(three.values()) == max(one.values())

    def test_an_activated_readout_keeps_the_class_scores_on_a_spiking_core(self):
        bare = NarrowConvNet((1, 28, 28), 10)
        activated = NarrowConvNet((1, 28, 28), 10, readout=ACTIVATED_READOUT)
        assert len(bare.head) == 1
        assert isinstance(activated.head[-1], torch.nn.ReLU)
        assert activated(torch.rand(2, 1, 28, 28)).min() >= 0.0

    def test_an_unknown_readout_is_refused(self):
        with pytest.raises(ValueError, match="readout="):
            NarrowConvNet((1, 28, 28), 10, readout="softmax")

    def test_a_body_stage_that_cannot_reduce_further_is_refused(self):
        with pytest.raises(ValueError, match="cannot reduce further"):
            NarrowConvNet((1, 4, 4), 10, stem_stride=4, body_blocks=3)

    def test_a_trunkless_build_is_refused(self):
        with pytest.raises(ValueError, match="out of range"):
            NarrowConvNet((1, 28, 28), 10, trunk_blocks=0)

    def test_a_non_chw_input_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            NarrowConvNet((28, 28), 10)

    def test_end_to_end_under_both_placements(self):
        model = NarrowConvNet((1, 28, 28), 10).eval()
        model(torch.rand(1, 1, 28, 28))
        for placement in ("subsume", "offload"):
            report = streamed_span_report_model(
                model, (1, 28, 28), 10, encoding_placement=placement,
            )
            assert report.end_to_end, placement

    def test_registered_builder_builds(self):
        builder_cls = ModelRegistry.get_builder_cls("narrow_conv")
        builder = builder_cls("cpu", (1, 28, 28), 10, {})
        model = builder.build({}).eval()
        assert model(torch.rand(1, 1, 28, 28)).shape == (1, 10)

    def test_the_builder_honors_its_configuration(self):
        builder_cls = ModelRegistry.get_builder_cls("narrow_conv")
        builder = builder_cls("cpu", (1, 28, 28), 10, {})
        model = builder.build({"stem_channels": 14, "stem_kernel": 13,
                               "stem_stride": 10, "body_blocks": 0,
                               "trunk_width": 120, "trunk_blocks": 2,
                               "readout": ACTIVATED_READOUT})
        assert model.stem[0].out_channels == 14
        assert model.stem[0].stride == (10, 10)
        assert model.head[0].in_channels == 120
        assert isinstance(model.head[-1], torch.nn.ReLU)
        assert max(_fan_ins(model).values()) <= 127
