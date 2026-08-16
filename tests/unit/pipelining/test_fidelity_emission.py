"""The C4 fidelity emission wiring: searched runs write, others write nothing."""

from types import SimpleNamespace

from mimarsinan.pipelining.pipeline_steps.verification.fidelity_emission import (
    _fidelity_axis_names,
    emit_run_fidelity,
)


def _pipeline(config):
    return SimpleNamespace(config=config, working_directory="/nonexistent")


class TestTheEmissionGate:
    def test_a_fixed_run_writes_nothing(self):
        """No search, no prediction, no correlation — None, not an empty file."""
        config = {"model_config_mode": "user", "hw_config_mode": "user"}
        assert emit_run_fidelity(_pipeline(config), record=None) is None

    def test_a_searched_run_without_declared_objectives_writes_nothing(self):
        config = {
            "model_config_mode": "user", "hw_config_mode": "search",
            "arch_search": {},
        }
        assert emit_run_fidelity(_pipeline(config), record=None) is None


class TestTheAxisFilter:
    def test_the_training_proxy_is_dropped(self):
        """The rebuild never trains; hardware completeness is what a deployed
        config can answer."""
        names = _fidelity_axis_names({
            "arch_search": {
                "objectives": ["estimated_accuracy", "chip_area_mm2"],
            },
        })
        assert names == ["chip_area_mm2"]

    def test_a_proxy_only_search_yields_no_axes(self):
        names = _fidelity_axis_names({
            "arch_search": {"objectives": ["estimated_accuracy"]},
        })
        assert names is None
