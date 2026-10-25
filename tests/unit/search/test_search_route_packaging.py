"""Sign-off M10, option (a): the joint search route builds and converts the candidate model the way the deployment
does (the plan's packaging contract and leaf decomposition), so the program the search prices is the program the
pipeline deploys. Before the fix the route built the value-domain LeNet-5 of t0_41 and t0_44 under the spiking
contract: 198 softcores against the deployed 983."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING, packaging_contract_for
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_problem import build_search_problem
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.problems.joint.model_build import DeployedConversion

from unit.mapping.residency.executed import templates as TV


def _problem(config: dict) -> JointArchHwProblem:
    problem, _names = build_search_problem(
        SimpleNamespace(config=config, data_provider_factory=None), search_mode="hardware",
        builder_cls=ModelRegistry.get_builder_cls(str(config["model_type"])), arch_options=[],
        model_config_assembler=lambda raw: dict(raw), seed=0,
    )
    return problem


def _searched_softcores(problem: JointArchHwProblem):
    """The softcores the search prices for its base candidate (``candidate_layout``'s model and walk)."""
    resolved = problem._resolved_configuration({"model_config": {}, "platform_constraints": {}})
    pcfg = resolved["platform_constraints"]
    placement = problem.candidate_encoding_placement(resolved)
    model, _total = problem._candidate_model(resolved["model_config"], pcfg, placement)
    return problem._collect_softcores(problem._ensure_mapper_repr(model, placement), pcfg)[0]


def _shapes(specs):
    return [(int(s.input_count), int(s.output_count), s.segment_id, s.bank_id) for s in specs]


@pytest.mark.parametrize("template", ["t0_41", "t0_44"])
def test_the_search_prices_the_program_the_pipeline_deploys(template):
    vehicle = TV.template_vehicle(template)
    problem = _problem(vehicle.config)
    assert problem.conversion.packaging == packaging_contract_for(DeploymentPlan.resolve(vehicle.config))
    assert problem.conversion.packaging.is_value_domain
    deployed = TV.layout_walk(vehicle)[0]
    searched = _searched_softcores(problem)
    print(f"[{template}] search route {len(searched)} softcores, deployed {len(deployed)}")
    assert len(searched) == len(deployed) == 983
    assert _shapes(searched) == _shapes(deployed)
    legacy = dataclasses.replace(problem, conversion=DeployedConversion())    # the route before the fix
    assert len(_searched_softcores(legacy)) == 198


def test_a_spiking_run_keeps_the_spiking_contract():
    problem = _problem(TV.template_vehicle("t0_53").config)
    assert problem.conversion.packaging == SPIKING_PACKAGING


def test_a_problem_built_without_a_run_declares_todays_default():
    fields = JointArchHwProblem.__dataclass_fields__
    assert fields["conversion"].default_factory().packaging == SPIKING_PACKAGING
    assert MVM_PACKAGING != SPIKING_PACKAGING


def test_the_core_semantics_is_not_a_searchable_option():
    """It selects the packaging contract and the pipeline's steps, both fixed when the problem is built, so a
    searched value would price candidates under a contract the winner does not deploy."""
    with pytest.raises(ValueError, match="core_semantics.*not searchable"):
        build_option_axes(["core_semantics"])
