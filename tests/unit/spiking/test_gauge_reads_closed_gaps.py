"""[U-TGAUGE5] Tests of U-TGAUGE4's gauge reads that its fourth verify found without one (its m-1 and m-2, mutants
X11 and X16) and the design review's correction 12 (mutant X12): each read is right on U-TGAUGE4's tree; these cases
pin it.

- The fold's per-input grid branch: a source vector of one gauge per element that holds one gauge per input channel
  (each channel's elements alike, the channels not one value) folds per input channel, read along the weight's input
  axis (a linear's last axis, a convolution's leading channel axis). Read the other way it refuses or folds the wrong
  channel.
- The bare read of a packed core: a per-core vector longer than the core's used neurons declares no resident's channel
  per neuron, so the read refuses by name at the first output neuron (a read that took the vector while it was longer
  than the neuron index placed it).
- ``holds_channels``: a host op whose vector holds one gauge per channel of its own output (its length the declared
  channel count) holds channels; a vector of one gauge per element does not (the closed form then would misplace it).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.hybrid_run.output_decode import OutputGaugeError, bare_mapping_output_gauge
from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource
from mimarsinan.mapping.support.gauge_placement import fold_per_input, holds_channels


def _linear(in_features):
    return types.SimpleNamespace(name="reader", layer=types.SimpleNamespace(weight=torch.zeros(3, in_features)))


def _conv(in_channels):
    return types.SimpleNamespace(name="reader", layer=types.SimpleNamespace(weight=torch.zeros(3, in_channels, 3, 3)))


def test_a_linear_over_a_token_join_folds_each_input_channel_at_its_producer_s_gauge():
    """Two tokens of three features, each feature's gauge alike over the tokens (the token-join of two producers at one
    per-channel gauge, laid token by token): the fold is the per-channel gauge, read along the linear's last axis."""
    channel = torch.tensor([0.5, 2.0, 1.25])
    per_element = torch.cat([channel, channel])          # (tokens, features) in C order
    assert torch.equal(fold_per_input(_linear(3), per_element, 3), channel)


def test_a_convolution_folds_each_input_channel_along_its_leading_axis():
    """A convolution's input channels lead its sample (channel-first): each channel's block of positions alike."""
    channel = torch.tensor([0.5, 2.0, 1.25])
    per_element = channel.repeat_interleave(4)           # (channels, positions) in C order
    assert torch.equal(fold_per_input(_conv(3), per_element, 3), channel)


def test_a_per_core_vector_longer_than_the_used_neurons_refuses_at_the_first_neuron():
    """Six values over three used neurons, its first three alike: no gauge per used neuron is declared, so the read
    refuses by name at neuron 0, where a read of the vector by the neuron index would take 2.0 for every output."""
    core = types.SimpleNamespace(activation_scale=torch.tensor([2.0, 2.0, 2.0, 5.0, 7.0, 9.0]),
                                 neurons_per_core=4, available_neurons=1)
    mapping = types.SimpleNamespace(cores=[core],
                                    output_sources=np.array([SpikeSource(0, i) for i in range(3)], dtype=object))
    with pytest.raises(OutputGaugeError, match="6-value activation scale over 3 used neurons"):
        bare_mapping_output_gauge(mapping)


def test_a_host_op_holds_channels_only_where_its_vector_is_one_per_channel_of_its_output():
    root = types.SimpleNamespace(get_source_mappers=lambda: [], output_shape=(3, 2, 2))
    op = types.SimpleNamespace(perceptron=None, get_source_mappers=lambda: [root], output_shape=(3, 2, 2),
                               input_shapes=[(3, 2, 2)])
    assert holds_channels(op, torch.ones(3)) is True
    assert holds_channels(op, torch.ones(12)) is False
