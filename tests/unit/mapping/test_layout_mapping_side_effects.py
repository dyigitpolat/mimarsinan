"""Shape-only layout walks must leave no IR-cache state on the live model (t0_20 pickle crash)."""

from __future__ import annotations

import pickle

import pytest
import torch

from mimarsinan.models.builders import BUILDERS_REGISTRY


def _built_simple_mlp():
    builder = BUILDERS_REGISTRY["simple_mlp"]("cpu", (1, 28, 28), 10, {})
    model = builder.build({"mlp_width_1": 64, "mlp_width_2": 32})
    with torch.no_grad():
        model.train()
        model(torch.randn(2, 1, 28, 28))
    return model


def _mapper_nodes(repr_):
    repr_._ensure_exec_graph()
    return list(repr_._exec_order)


def test_collect_layout_softcores_leaves_model_picklable() -> None:
    """The einops-flow layout walk caches LayoutSourceView closures in
    ``Mapper._ir_sources``; a shape-only caller (wizard/search/GUI snapshot)
    must clear them or the next pipeline model save dies un-picklable."""
    from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping

    model = _built_simple_mlp()
    pickle.dumps(model)  # sanity: picklable before

    layout = LayoutIRMapping(max_axons=256, max_neurons=256)
    softcores = layout.collect_layout_softcores(model.get_mapper_repr())
    assert softcores

    pickle.dumps(model)  # the regression pin: still picklable after
    assert all(
        getattr(node, "_ir_sources", None) is None
        for node in _mapper_nodes(model.get_mapper_repr())
    )


def test_snapshot_planned_mapping_is_side_effect_free() -> None:
    """The GUI step-snapshot estimate path is telemetry; it must not poison
    the model for the next cache save (the t0_20 Pretraining crash)."""
    from mimarsinan.mapping.verification.wizard_layout_verify import (
        verify_planned_mapping_performance,
    )

    model = _built_simple_mlp()
    constraints = {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "max_axons": 256,
        "max_neurons": 256,
    }
    result = verify_planned_mapping_performance(model.get_mapper_repr(), constraints)
    assert result is not None
    pickle.dumps(model)
