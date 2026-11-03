"""[U-LMP, U-BRG review m-5] The bias-grid dominance advisory states the raw-unit ratio it reads.

The advisory runs at Torch Mapping, before any grid exists, and reads ``max|effective bias| / max|effective weight|``
with the always-on row at 1.0. It said "the bias row sets the grid" unconditionally, which is false on the integer value
path with the bias on a row: Weight Quantization drives the row at its reader's input full scale ``x_fs`` there, so the
shared grid compares ``max|w|`` with ``max|b| / x_fs`` (W38 ResNet-50 arm N printed the sentence at 21.9 > 3.5). The
detail now names the ratio as raw units, and the grid rule with the drive. When it fires is unchanged.
"""

from __future__ import annotations

from mimarsinan.advisories.engine import evaluate_graph_advisories
from unit.advisories import test_rules_graph as RG  # a module: its Test class is not collected again here

ADVISORY = "ADV-BIAS-GRID-DOMINANCE"


def _dominated():
    return RG.TestBiasGridDominance()._dominated_repr()


def _drive_route_config() -> dict:
    return {"core_semantics": "mvm", "weight_quantization": True, "weight_bits": 5, "activation_bits": 8,
            "readout_law": {"adc_bits": 32, "mult_bits": 24, "range": "calibrated"}, "has_bias": False,
            "wq_two_scale_projection": False}


def _detail(model_repr, config) -> str:
    fired = [a for a in evaluate_graph_advisories(model_repr, config) if a.id == ADVISORY]
    assert len(fired) == 1, f"{ADVISORY} fired {len(fired)} times"
    return fired[0].detail


def test_the_advisory_states_the_raw_unit_ratio_and_the_drive_s_grid_rule():
    """Red before: the detail said the bias row sets the grid without naming the gauge it read, on every route."""
    for config in (_drive_route_config(), RG.TestBiasGridDominance()._wq_config()):
        detail = _detail(_dominated(), config)
        assert "in raw units, the always-on row read at 1.0" in detail
        assert "max|b| / x_fs" in detail and "input full scale" in detail
        assert "where the row is read at 1.0 the bias row sets the grid" in detail


def test_the_advisory_fires_as_before_on_the_drive_route_and_stays_silent_on_healthy_ratios():
    """Guard: the trigger is unchanged (the dominated fixture fires on the drive route too; healthy ratios are silent)."""
    assert _detail(_dominated(), _drive_route_config())
    assert ADVISORY not in {a.id for a in evaluate_graph_advisories(RG._converted_deep_mlp(depth=3),
                                                                    _drive_route_config())}
