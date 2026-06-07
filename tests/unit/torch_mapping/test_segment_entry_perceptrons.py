"""segment_entry_perceptrons: first on-chip perceptron of each neural segment."""

import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter as _CA
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import EinopsRearrangeMapper, InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons


def _perceptron(out_f, in_f):
    return Perceptron(out_f, in_f, normalization=nn.Identity(),
                      base_activation_name="ReLU")


def _chain(*, mark_first_host=False, with_compute_op=False):
    p1 = _perceptron(6, 16)
    p2 = _perceptron(4, 6)
    p1.is_encoding_layer = bool(mark_first_host)
    p2.is_encoding_layer = False

    inp = InputMapper((1, 4, 4))
    flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
    m1 = PerceptronMapper(flat, p1)
    src = m1
    if with_compute_op:
        src = ComputeOpMapper(m1, _CA(torch.relu))
    m2 = PerceptronMapper(src, p2)
    return ModelRepresentation(m2), p1, p2


class TestSegmentEntryPerceptrons:
    def test_offload_style_first_perceptron_is_entry(self):
        repr_, p1, p2 = _chain()
        entries = segment_entry_perceptrons(repr_)
        assert entries == [p1], (
            "with no host encoder, the raw-input-fed perceptron starts segment 0"
        )

    def test_subsume_style_entry_is_first_on_chip_after_host_encoder(self):
        repr_, p1, p2 = _chain(mark_first_host=True)
        entries = segment_entry_perceptrons(repr_)
        assert entries == [p2], (
            "a host encoding layer is not on-chip; the q(x) seam is the NEXT "
            "perceptron's input (the value entering the first NeuralCore)"
        )

    def test_compute_op_starts_a_new_segment(self):
        repr_, p1, p2 = _chain(with_compute_op=True)
        entries = segment_entry_perceptrons(repr_)
        assert entries == [p1, p2], (
            "a ComputeOp is a hybrid stage barrier; its consumer perceptron "
            "reads a freshly grid-snapped stage input"
        )

    def test_structural_mappers_are_transparent(self):
        # The Einops between input and p1 must not hide the entry.
        repr_, p1, _ = _chain()
        assert p1 in segment_entry_perceptrons(repr_)
