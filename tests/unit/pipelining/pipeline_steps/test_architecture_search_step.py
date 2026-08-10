"""ArchitectureSearchStep hardware-mode smoke test.

Hardware-only search must run end-to-end on the resolved platform base (the
historical failure: ``fixed_platform_constraints`` stayed ``None`` outside
model mode, every candidate died on ``KeyError: 'cores'``, and the step
reported a misleading "no candidates" error).
"""

from conftest import MockPipeline, default_config

from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_step import (
    ArchitectureSearchStep,
)


def _hardware_search_config():
    cfg = default_config()
    cfg.update({
        "model_type": "simple_mlp",
        "model_config": {
            "mlp_width_1": 16,
            "mlp_width_2": 16,
            "base_activation": "ReLU",
        },
        "hw_config_mode": "search",
        "weight_bits": 4,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "arch_search": {
            "optimizer": "nsga2",
            "pop_size": 4,
            "generations": 2,
            "seed": 0,
            "num_core_types": 1,
            "core_axons_bounds": [64, 256],
            "core_neurons_bounds": [64, 256],
            "core_count_bounds": [8, 64],
        },
    })
    return cfg


def _run_step(tmp_path):
    pipeline = MockPipeline(
        config=_hardware_search_config(),
        working_directory=str(tmp_path / "search_step"),
    )
    step = ArchitectureSearchStep(pipeline)
    step.name = "ArchitectureSearch"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


class TestHardwareModeSmoke:
    def test_step_completes_and_promises_entries(self, tmp_path):
        pipeline = _run_step(tmp_path)
        cache = pipeline.cache
        assert "ArchitectureSearch.model_config" in cache
        assert "ArchitectureSearch.model_builder" in cache
        assert "ArchitectureSearch.platform_constraints_resolved" in cache
        assert "ArchitectureSearch.architecture_search_result" in cache

        result = cache["ArchitectureSearch.architecture_search_result"]
        assert result["search_mode_used"] == "hardware"
        assert result["best"]["configuration"], "search must land a best candidate"
        assert result["discovered_platform_constraints"] is not None

    def test_resolved_platform_is_base_plus_searched_cores(self, tmp_path):
        pipeline = _run_step(tmp_path)
        pcfg = pipeline.cache["ArchitectureSearch.platform_constraints_resolved"]

        assert pcfg["cores"], "searched cores must be present"
        for core in pcfg["cores"]:
            assert 64 <= core["max_axons"] <= 256
            assert 64 <= core["max_neurons"] <= 256
            assert 8 <= core["count"] <= 64
            assert core["has_bias"] is True

        # The deployed base rides along: search and deployment see one chip.
        assert pcfg["weight_bits"] == 4, "config weight_bits, not a hardcoded 8"
        assert pcfg["allow_scheduling"] is True
        assert pcfg[CANONICAL_KEY] is False
        assert "schedule_policy" in pcfg
        assert "max_schedule_passes" in pcfg

    def test_fixed_model_config_is_passed_through(self, tmp_path):
        pipeline = _run_step(tmp_path)
        model_config = pipeline.cache["ArchitectureSearch.model_config"]
        assert model_config == _hardware_search_config()["model_config"]

    def test_best_objectives_are_not_penalties(self, tmp_path):
        pipeline = _run_step(tmp_path)
        result = pipeline.cache["ArchitectureSearch.architecture_search_result"]
        best_objectives = result["best"]["objectives"]
        assert best_objectives, "hardware mode has layout-proxy objectives"
        for name, value in best_objectives.items():
            assert abs(float(value)) < 1e17, f"{name} looks like a penalty: {value}"
