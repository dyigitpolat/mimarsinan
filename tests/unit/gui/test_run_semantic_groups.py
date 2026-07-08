"""Saved run documents are sparse; every executed step still gets its phase.

``config.json`` records only the keys the user set, so resolving a pipeline plan
straight from it silently drops every defaulted step (weight quantization, the
simulation backends). The monitor paints step-timing bars and navigator dots
from ``semantic_group``, so a dropped step is rendered as the ``other`` phase —
a lie. The display view already resolves defaults + derivations, so its
``pipeline_preview`` is the single source of truth for the phase of a step.
"""

import json
import os

import pytest

from mimarsinan.gui.runtime.persistence import save_step_to_persisted
from mimarsinan.gui.runtime.persistence.store import save_run_info
from mimarsinan.gui.runs import get_run_pipeline
from mimarsinan.gui.runtime.process_monitor import get_run_detail
from mimarsinan.gui.runtime.process_spawn import ManagedRun
from mimarsinan.gui.viewmodel import semantic_groups_from_config_view

# A realistic *sparse* saved document: neither `weight_quantization` nor any
# simulation backend appears, yet all of them run (they default on).
SPARSE_SAVED_CONFIG = {
    "seed": 0,
    "experiment_name": "sparse_cfg",
    "generated_files_path": "./generated",
    "data_provider_name": "MNIST_DataProvider",
    "platform_constraints": {
        "cores": [{"max_axons": 784, "max_neurons": 512, "count": 60, "has_bias": True}],
        "has_bias": True,
        "target_tq": 16,
        "simulation_steps": 16,
        "weight_bits": 5,
    },
    "deployment_parameters": {
        "model_type": "lenet5",
        "model_config": {"variant": "lenet5"},
        "lr": 0.003,
        "training_epochs": 1,
        "batch_size": 128,
        "max_simulation_samples": 4,
    },
    "start_step": None,
}

# Steps that only exist once defaults/derivations are applied.
DEFAULTED_STEPS = ("Weight Quantization", "Quantization Verification",
                   "Core Quantization Verification", "SANA-FE Simulation")


def _write_run(run_dir, step_names):
    os.makedirs(os.path.join(run_dir, "_RUN_CONFIG"), exist_ok=True)
    with open(os.path.join(run_dir, "_RUN_CONFIG", "config.json"), "w", encoding="utf-8") as f:
        json.dump(SPARSE_SAVED_CONFIG, f)
    for i, name in enumerate(step_names):
        save_step_to_persisted(
            run_dir, step_name=name, start_time=float(i), end_time=float(i) + 1.0,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
        )


class TestSemanticGroupsFromConfigView:
    def test_maps_preview_steps_to_their_groups(self) -> None:
        view = {"pipeline_preview": {"steps": ["A", "B"], "semantic_groups": ["activation", "hardware"]}}
        assert semantic_groups_from_config_view(view) == {"A": "activation", "B": "hardware"}

    @pytest.mark.parametrize("view", [None, {}, {"pipeline_preview": None},
                                      {"pipeline_preview": {"steps": [], "semantic_groups": []}}])
    def test_degrades_to_empty_map(self, view) -> None:
        assert semantic_groups_from_config_view(view) == {}


class TestHistoricalRunSemanticGroups:
    def test_defaulted_steps_are_labelled(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = tmp_path / "run_sparse"
        _write_run(str(run_dir), ("Pretraining", *DEFAULTED_STEPS))

        pipeline = get_run_pipeline("run_sparse")
        groups = {s["name"]: s["semantic_group"] for s in pipeline["steps"]}

        unlabelled = [n for n, g in groups.items() if g is None]
        assert not unlabelled, f"steps rendered as the 'other' phase: {unlabelled}"
        assert groups["Weight Quantization"] == "weight_quantization"
        assert groups["SANA-FE Simulation"] == "simulation"
        assert groups["Core Quantization Verification"] == "core_verification"


class TestActiveRunSemanticGroups:
    def test_defaulted_steps_are_labelled(self, tmp_path) -> None:
        run_dir = tmp_path / "active_sparse"
        step_names = ["Pretraining", *DEFAULTED_STEPS]
        _write_run(str(run_dir), step_names)
        save_run_info(str(run_dir), pid=os.getpid(), step_names=step_names)

        managed = ManagedRun(run_id="active_sparse", working_dir=str(run_dir),
                             pid=os.getpid(), started_at=0.0)
        detail = get_run_detail({"active_sparse": managed}, "active_sparse")
        assert detail is not None
        groups = {s["name"]: s["semantic_group"] for s in detail["steps"]}

        unlabelled = [n for n, g in groups.items() if g is None]
        assert not unlabelled, f"steps rendered as the 'other' phase: {unlabelled}"
        assert groups["Weight Quantization"] == "weight_quantization"
        assert groups["SANA-FE Simulation"] == "simulation"
