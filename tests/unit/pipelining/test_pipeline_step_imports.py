"""Pipeline steps import from subpackages and match deployment registration."""

from __future__ import annotations

import importlib

import pytest

from mimarsinan.pipelining.pipeline_steps import (
    ArchitectureSearchStep,
    HardCoreMappingStep,
    SanafeSimulationStep,
    SoftCoreMappingStep,
)

_SUBPACKAGE_PATHS = {
    ArchitectureSearchStep: "mimarsinan.pipelining.pipeline_steps.config.architecture_search_step",
    SoftCoreMappingStep: "mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step",
    HardCoreMappingStep: "mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step",
    SanafeSimulationStep: "mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step",
}


@pytest.mark.parametrize("cls,mod_path", list(_SUBPACKAGE_PATHS.items()))
def test_step_class_lives_in_subpackage(cls, mod_path):
    mod = importlib.import_module(mod_path)
    assert getattr(mod, cls.__name__) is cls


def test_deployment_pipeline_registers_steps():
    from mimarsinan.pipelining.pipelines.deployment_pipeline import get_pipeline_step_specs

    specs_default = get_pipeline_step_specs({})
    step_types = {cls for _, cls in specs_default}
    assert SoftCoreMappingStep in step_types

    specs_sanafe = get_pipeline_step_specs({"enable_sanafe_simulation": True})
    sanafe_types = {cls for _, cls in specs_sanafe}
    assert SanafeSimulationStep in sanafe_types
