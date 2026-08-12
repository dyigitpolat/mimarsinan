"""The GUI's PLANNED mapping panel must plan the mapping the run will deploy.

Fourth instance of this unit's defect class. ``snapshot_mapping_performance_planned``
converts a pre-conversion torch module to get a layout to display, and it
converted with ``convert_torch_model``'s DEFAULT placement — so during an
``offload`` run, between Model Building and Torch Mapping, the panel showed the
subsumed mapping: a different core count for a chip the run was never going to
build. The user watches that panel to decide whether the platform fits.
"""

from __future__ import annotations

import pytest

from mimarsinan.gui.snapshot.mapping_snapshot import (
    snapshot_mapping_performance_planned,
)
from mimarsinan.models.lenet5 import LeNet5

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
PLATFORM = {
    "cores": [{"max_axons": 784, "max_neurons": 256, "count": 256}],
    "max_axons": 784,
    "max_neurons": 256,
    "allow_coalescing": True,
}


def _planned(placement: str):
    return snapshot_mapping_performance_planned(
        LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES),
        dict(PLATFORM),
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        encoding_placement=placement,
    )


class TestPlannedMappingHonorsPlacement:
    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_it_produces_a_panel_for_either_placement(self, placement):
        assert _planned(placement) is not None

    def test_the_two_placements_plan_different_chips(self):
        subsumed, offloaded = _planned("subsume"), _planned("offload")
        assert subsumed["total_softcores"] < offloaded["total_softcores"], (
            f"planned layout ignored the placement: "
            f"{subsumed['total_softcores']} vs {offloaded['total_softcores']}"
        )

    def test_a_model_that_is_already_a_flow_keeps_its_own_resolved_placement(self):
        """A built flow carries the deployment's marking; the panel reads it,
        it does not re-resolve (re-marking is what the late gate must never do)."""
        from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
        from mimarsinan.torch_mapping.encoding_layers import (
            resolved_encoding_placement,
        )

        builder = BUILDERS_REGISTRY["simple_mlp"]("cpu", INPUT_SHAPE, NUM_CLASSES, {})
        flow = build_model(
            builder, {"mlp_width_1": 64, "mlp_width_2": 32},
            encoding_placement="offload",
        )
        assert snapshot_mapping_performance_planned(
            flow, dict(PLATFORM), input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
            encoding_placement="subsume",
        ) is not None
        assert resolved_encoding_placement(flow.get_mapper_repr()) == "offload"


def test_the_snapshot_builder_passes_the_configured_placement():
    """Wiring: the panel's placement comes from the run's config, not a default."""
    import ast
    import inspect

    from mimarsinan.gui.snapshot import builders

    tree = ast.parse(inspect.getsource(builders))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "snapshot_mapping_performance_planned"
    ]
    assert calls, "the planned-mapping snapshot is no longer called from builders"
    for call in calls:
        keyword = next(
            (k for k in call.keywords if k.arg == "encoding_placement"), None
        )
        assert keyword is not None, "planned mapping is planned under a default placement"
        assert "encoding_layer_placement" in ast.unparse(keyword.value), (
            "the placement must come from the run's config, not a literal"
        )
