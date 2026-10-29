"""CL2 x CL3 (integration of the exploration foundations): sign-off M10 on CL2's three target cells.

t0_66, t0_67 and t0_68 are the minimal pairs of t0_44 on the HERMES-64, SRAM-CIM macro and NeuRRAM bundles. On each,
the search route prices the program the pipeline deploys (M10: the value-domain LeNet-5's 983 softcores, where the
route before M10 built 198), at the cores the cell's target bundle declares (CL2)."""

from __future__ import annotations

import dataclasses

import pytest

from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.search.problems.joint.model_build import DeployedConversion

from unit.mapping.residency.executed import templates as TV
from unit.search.test_search_route_packaging import _problem, _searched_softcores, _shapes

#: (template, target, declared cores as (max_axons, max_neurons, count)): the bundle geometry; the macro's count is
#: the cell's target_overrides entry ("cores declared"); [U-C4] NeuRRAM's 256 x 256 cells hold 128 x 256 weights (two
#: cells per signed weight, along axons).
_CELLS = [
    ("t0_66", "hermes_64", [(256, 256, 64)]),
    ("t0_67", "sram_cim_macro", [(128, 1024, 8)]),
    ("t0_68", "neurram", [(128, 256, 48)]),
]


@pytest.mark.parametrize("template,target,cores", _CELLS, ids=[c[0] for c in _CELLS])
def test_the_search_prices_the_deployed_program_on_the_target_cell(template, target, cores):
    vehicle = TV.template_vehicle(template)
    assert vehicle.config["target"] == target
    assert [(c["max_axons"], c["max_neurons"], c["count"]) for c in vehicle.config["cores"]] == cores
    problem = _problem(vehicle.config)
    assert problem.conversion.packaging == packaging_contract_for(DeploymentPlan.resolve(vehicle.config))
    assert problem.conversion.packaging.is_value_domain
    deployed = TV.layout_walk(vehicle)[0]
    searched = _searched_softcores(problem)
    assert len(searched) == len(deployed) == 983
    assert _shapes(searched) == _shapes(deployed)
    assert len(_searched_softcores(dataclasses.replace(problem, conversion=DeployedConversion()))) == 198
