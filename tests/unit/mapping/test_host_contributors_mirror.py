"""The named host contributors must sum to the host total the refusal reports.

``count_host_params`` is the authoritative host census the on-chip floor gate
divides by; ``host_contributors_from_ir`` is the itemization printed next to it.
They only add up if they dedupe the same way — the itemizer keyed on the WRAPPED
PERCEPTRON while the census keys on the MODULE, so a perceptron reachable
through two host wrappers was counted twice in the total and named once in the
breakdown. A refusal whose parts do not sum to its own total is a refusal a
reader cannot act on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.support.host_contributors import host_contributors_from_ir
from mimarsinan.mapping.verification.onchip_majority import count_host_params
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10


class _StubOp:
    def __init__(self, module=None, op_type="module", bound_tensors=None):
        self.op_type = op_type
        self.params = {"module": module}
        if bound_tensors is not None:
            self.params["bound_tensors"] = bound_tensors


class _StubGraph:
    def __init__(self, ops):
        self._ops = ops

    def get_compute_ops(self):
        return self._ops


class _Wrapper(nn.Module):
    """A host ComputeOp module that wraps a perceptron, as the conv mapper does."""

    def __init__(self, perceptron):
        super().__init__()
        self.perceptron = perceptron


class _Perceptron(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.is_encoding_layer = True


def test_two_wrappers_of_one_perceptron_are_named_as_often_as_counted():
    """The divergence itself: the census sees two modules, so must the itemizer."""
    perceptron = _Perceptron()
    graph = _StubGraph([_StubOp(_Wrapper(perceptron)), _StubOp(_Wrapper(perceptron))])

    contributors = host_contributors_from_ir(graph)
    assert sum(unit.params for unit in contributors) == count_host_params(graph)
    assert len(contributors) == 2


def test_one_wrapper_seen_twice_is_still_counted_once():
    """Deduping is by module IDENTITY on both sides, not by occurrence."""
    wrapper = _Wrapper(_Perceptron())
    graph = _StubGraph([_StubOp(wrapper), _StubOp(wrapper)])

    contributors = host_contributors_from_ir(graph)
    assert len(contributors) == 1
    assert sum(unit.params for unit in contributors) == count_host_params(graph)


def test_the_wrapped_perceptrons_role_still_drives_the_label():
    """Keying on the module must not lose the encoder role read off the unit."""
    graph = _StubGraph([_StubOp(_Wrapper(_Perceptron()))])
    unit = host_contributors_from_ir(graph)[0]
    assert unit.is_encoder
    assert unit.label.startswith("subsumed encoding layer Linear")


def test_a_module_free_op_still_contributes_its_bound_constants():
    graph = _StubGraph([_StubOp(None, op_type="add", bound_tensors=[torch.zeros(5)])])
    contributors = host_contributors_from_ir(graph)
    assert [(u.label, u.params, u.is_encoder) for u in contributors] == [
        ("host op add constants", 5, False)
    ]
    assert sum(u.params for u in contributors) == count_host_params(graph)


def test_the_parts_sum_to_the_total_on_a_real_mapped_graph():
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    builder = BUILDERS_REGISTRY["simple_mlp"]("cpu", INPUT_SHAPE, NUM_CLASSES, {})
    model = build_model(
        builder, {"mlp_width_1": 64, "mlp_width_2": 32}, encoding_placement="subsume",
    )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *INPUT_SHAPE))
    repr_ = model.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir_graph = IRMapping(
        q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
        allow_coalescing=False, hardware_bias=True,
    ).map(repr_)

    contributors = host_contributors_from_ir(ir_graph)
    assert contributors
    assert sum(unit.params for unit in contributors) == count_host_params(ir_graph)
