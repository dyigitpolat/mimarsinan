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
#: A SECOND, differently-shaped host-majority vehicle. One model can be passed
#: by a canned string; two cannot — every number below has to be measured.
WIDE_MLP_CONFIG = {"mlp_width_1": 128, "mlp_width_2": 32, "base_activation": "ReLU"}


def _subsumed_mlp(model_config=None):
    builder = BUILDERS_REGISTRY["simple_mlp"](
        "cpu", INPUT_SHAPE, NUM_CLASSES, {"target_tq": 32, "device": "cpu"}
    )
    model = build_model(
        builder, model_config or MLP_CONFIG, encoding_placement="subsume",
    )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *INPUT_SHAPE))
    return model


def _static_message(model_config=None) -> str:
    model = _subsumed_mlp(model_config)
    with pytest.raises(OnchipMajorityError) as excinfo:
        assert_onchip_majority_estimate_or_raise(
            model, INPUT_SHAPE, NUM_CLASSES,
            encoding_placement="subsume", min_fraction=0.2,
        )
    return str(excinfo.value)


@pytest.fixture(scope="module")
def static_message():
    return _static_message()


@pytest.fixture(scope="module")
def narrow_message():
    return _static_message(WIDE_MLP_CONFIG)


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

    def test_it_offers_each_escape_hatch_exactly_once(self, static_message):
        """The placement remedy and the refusal both used to name the escapes."""
        assert static_message.count("onchip_majority_gate=false") == 1, static_message
        assert static_message.count("onchip_min_fraction") == 1, static_message


class TestASecondModelGetsItsOwnNumbers:
    """One vehicle can be satisfied by a canned string; two cannot.

    Every measured quantity below differs from the fixture model's, so a message
    that stopped measuring — a hardcoded parenthetical, a memoized contributor
    list, a role read off a label prefix — fails here.
    """

    def test_it_reports_this_models_own_split(self, narrow_message, static_message):
        assert "8.96%" in narrow_message, narrow_message
        assert "110658" in narrow_message, narrow_message
        assert "15.23%" not in narrow_message
        assert "237666" not in narrow_message
        assert narrow_message != static_message

    def test_it_names_this_models_own_encoder_with_its_shape_and_size(
        self, narrow_message
    ):
        assert "subsumed encoding layer Linear 784->128" in narrow_message
        assert "100742" in narrow_message, narrow_message
        assert "784->256" not in narrow_message

    def test_it_never_calls_a_host_op_offloaded(self, narrow_message):
        for match in re.finditer(r"offload\w*", narrow_message):
            window = narrow_message[max(0, match.start() - 60):match.end() + 60]
            assert "host" not in window.lower() or "on chip" in window.lower(), (
                f"'offload' used to describe host placement: ...{window}..."
            )

    def test_it_offers_each_escape_hatch_exactly_once(self, narrow_message):
        assert narrow_message.count("onchip_majority_gate=false") == 1
        assert narrow_message.count("onchip_min_fraction") == 1


class TestMappedGraphFloorMessage:
    """The IR-graph twin of the same refusal, from the same vocabulary."""

    def _message(self, model_config=None):
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

        model = _subsumed_mlp(model_config)
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

    def test_a_second_model_gets_its_own_measured_numbers(self):
        message = self._message(WIDE_MLP_CONFIG)
        assert "100742" in message, message
        assert "784->128" in message, message
        assert "201478" not in message

    def test_it_offers_each_escape_hatch_exactly_once(self):
        message = self._message()
        assert message.count("onchip_majority_gate=false") == 1, message
        assert message.count("onchip_min_fraction") == 1, message


class TestTheEscapeHatchesAreOwnedByTheRefusal:
    """One sentence names the escapes; the placement remedy never repeats it.

    The remedy has a branch for a host majority placement CANNOT move, and that
    branch used to end with the same "onchip_majority_gate=false / lower
    onchip_min_fraction" advice the refusal appends — so that message told the
    reader twice, in two different wordings.
    """

    def _hostile_graph(self):
        """A mapped graph whose host majority is NOT an encoder."""
        import torch.nn as nn

        class _Op:
            def __init__(self, module):
                self.op_type = "module"
                self.params = {"module": module}

        class _Graph:
            def __init__(self, ops):
                self._ops = ops

            def get_compute_ops(self):
                return self._ops

        readout = nn.Linear(64, 64)
        assert not getattr(readout, "is_encoding_layer", False)
        return _Graph([_Op(readout)]), int(sum(p.numel() for p in readout.parameters()))

    def _message_without_a_subsumed_encoder(self) -> str:
        graph, host_params = self._hostile_graph()
        with pytest.raises(OnchipMajorityError) as excinfo:
            assert_onchip_majority_or_raise(
                graph, total_params=host_params, min_fraction=0.2
            )
        return str(excinfo.value)

    def test_it_says_placement_cannot_help_and_offers_the_escapes_once(self):
        message = self._message_without_a_subsumed_encoder()
        assert "No encoding layer is subsumed here" in message, message
        assert message.count("onchip_majority_gate=false") == 1, message
        assert message.count("onchip_min_fraction") == 1, message

    def test_no_remedy_branch_names_an_escape_hatch(self):
        """Structural: the remedy is about PLACEMENT, the escapes are the caller's."""
        from mimarsinan.mapping.verification.onchip_fraction import (
            OnchipFractionEstimate,
            _placement_remedy,
        )
        from mimarsinan.mapping.support.host_contributors import HostUnit
        from mimarsinan.mapping.verification.onchip_majority import (
            OnchipParamBreakdown,
            onchip_placement_remedy,
        )

        encoder = [HostUnit("subsumed encoding layer Linear 4->4", 90, True)]
        plain = [HostUnit("host op Linear 4->4", 90, False)]
        breakdown = OnchipParamBreakdown(
            onchip_params=10, host_params=90, total_params=100
        )
        remedies = [
            onchip_placement_remedy(encoder, breakdown),
            onchip_placement_remedy(plain, breakdown),
            onchip_placement_remedy(plain, OnchipParamBreakdown(0, 0, 0)),
        ]
        for placement in ("subsume", "offload"):
            for metric in ("params", "macs"):
                remedies.append(
                    _placement_remedy(
                        plain,
                        OnchipFractionEstimate(
                            onchip=10, host=90, total=100,
                            metric=metric, placement=placement,
                        ),
                    )
                )
        for remedy in remedies:
            assert "onchip_majority_gate" not in remedy, remedy
            assert "onchip_min_fraction" not in remedy, remedy


class TestTheEncoderRoleIsStructural:
    """``subsumed_encoder_params`` must read the ROLE, not a label prefix.

    Keying the remedy off ``label.startswith("subsumed")`` makes a human-readable
    string load-bearing: reword the label and the refusal silently starts
    claiming ``offload`` can do nothing (or everything).
    """

    def _contributors(self, model):
        from mimarsinan.mapping.verification.onchip_fraction import _host_unit
        from mimarsinan.mapping.support.host_contributors import (
            host_contributors_from_flow,
        )

        return host_contributors_from_flow(model, _host_unit)

    def test_the_encoder_carries_a_role_flag_of_its_own(self):
        contributors = self._contributors(_subsumed_mlp())
        assert [unit.is_encoder for unit in contributors].count(True) == 1
        encoder = next(unit for unit in contributors if unit.is_encoder)
        assert encoder.params == 201478

    def test_the_reachable_fraction_survives_a_relabelled_unit(self):
        from mimarsinan.mapping.support.host_contributors import (
            HostUnit,
            subsumed_encoder_params,
        )

        contributors = self._contributors(_subsumed_mlp())
        relabelled = [
            HostUnit("a completely different wording", unit.params, unit.is_encoder)
            for unit in contributors
        ]
        assert subsumed_encoder_params(relabelled) == subsumed_encoder_params(
            contributors
        ) == 201478

    def test_a_non_encoder_host_unit_is_not_counted_as_one(self):
        from mimarsinan.mapping.support.host_contributors import (
            HostUnit,
            subsumed_encoder_params,
        )

        units = [
            HostUnit("subsumed encoding layer Linear 4->4", 100, False),
            HostUnit("host op Linear 4->4", 7, True),
        ]
        assert subsumed_encoder_params(units) == 7
