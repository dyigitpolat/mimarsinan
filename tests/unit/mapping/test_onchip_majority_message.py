"""The on-chip-majority refusals must describe THIS model, in the project's vocabulary.

Both floor messages used to end with a canned parenthetical — "The host-side
ComputeOps (offloaded encoding Linear/Conv, classifier readout, attention) hold
the parameter majority" — which is wrong twice on a plain MLP: it names
``attention`` for a model that has none, and it says "offloaded" to mean "moved
to the host", the exact OPPOSITE of the ``placement='offload'`` value printed in
the same sentence. A reader following that message would flip the knob the wrong
way.

The vocabulary (config_schema/registry/entries_conversion.py):
    subsume = the encoder runs HOST-side; offload = the encoding layer is
    mapped ON CHIP.
"""

from __future__ import annotations

import re

import pytest
import torch

from mimarsinan.mapping.verification.onchip_fraction import (
    assert_onchip_majority_estimate_or_raise,
)
from mimarsinan.mapping.verification.onchip_majority import (
    OnchipMajorityError,
    assert_onchip_majority_or_raise,
)
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
MLP_CONFIG = {"mlp_width_1": 256, "mlp_width_2": 64, "base_activation": "ReLU"}


def _subsumed_mlp():
    builder = BUILDERS_REGISTRY["simple_mlp"](
        "cpu", INPUT_SHAPE, NUM_CLASSES, {"target_tq": 32, "device": "cpu"}
    )
    model = build_model(builder, MLP_CONFIG, encoding_placement="subsume")
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *INPUT_SHAPE))
    return model


@pytest.fixture(scope="module")
def static_message():
    model = _subsumed_mlp()
    with pytest.raises(OnchipMajorityError) as excinfo:
        assert_onchip_majority_estimate_or_raise(
            model, INPUT_SHAPE, NUM_CLASSES,
            encoding_placement="subsume", min_fraction=0.2,
        )
    return str(excinfo.value)


class TestStaticFloorMessage:
    def test_it_reports_the_measured_split(self, static_message):
        assert "15.23%" in static_message or re.search(r"\d+\.\d+%", static_message)
        assert "subsume" in static_message

    def test_it_never_calls_a_host_op_offloaded(self, static_message):
        """``offload`` means ON CHIP; using it for the host inverts the remedy."""
        for match in re.finditer(r"offload\w*", static_message):
            window = static_message[max(0, match.start() - 60):match.end() + 60]
            assert "host" not in window.lower() or "on chip" in window.lower(), (
                f"'offload' used to describe host placement: ...{window}..."
            )

    def test_it_names_no_op_this_model_does_not_have(self, static_message):
        assert "attention" not in static_message.lower()
        assert "conv" not in static_message.lower()

    def test_it_names_the_actual_top_host_contributor_with_its_size(self, static_message):
        """The encoder perceptron IS the host majority — say so, with its count."""
        assert "201478" in static_message, static_message

    def test_it_states_the_remedy_with_the_number_it_would_reach(self, static_message):
        assert "offload" in static_message
        assert re.search(r"100\.00%|100%", static_message), static_message


class TestMappedGraphFloorMessage:
    """The IR-graph twin of the same refusal, from the same vocabulary."""

    def _message(self):
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

        model = _subsumed_mlp()
        repr_ = model.get_mapper_repr()
        repr_.assign_perceptron_indices()
        compute_per_source_scales(repr_)
        ir_graph = IRMapping(
            q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
            allow_coalescing=False, hardware_bias=True,
        ).map(repr_)
        total = int(sum(p.numel() for p in model.parameters()))
        with pytest.raises(OnchipMajorityError) as excinfo:
            assert_onchip_majority_or_raise(
                ir_graph, total_params=total, min_fraction=0.2
            )
        return str(excinfo.value)

    def test_it_never_calls_a_host_op_offloaded(self):
        message = self._message()
        for match in re.finditer(r"offload\w*", message):
            window = message[max(0, match.start() - 60):match.end() + 60]
            assert "host" not in window.lower() or "on chip" in window.lower(), (
                f"'offload' used to describe host placement: ...{window}..."
            )

    def test_it_names_no_op_this_model_does_not_have(self):
        assert "attention" not in self._message().lower()

    def test_it_names_the_actual_top_host_contributors(self):
        message = self._message()
        assert "201478" in message, message
