"""[Residency unit, fix round] The candidate plane's refusal by name, recorded beside the record and plan planes.

K12 (route (b)'s census check) reads the record and plan planes of an executed residency program at its executed
step; the candidate (shape-only) plane is printed beside them and is not part of its verdict. The shape-only plane
refuses a declared plan by name where its own program differs from the builder's: on route (b)'s C1 at 256 x 64 under
B5i's index 10 its walk census states no live set, so its step function is one step and index 10 is outside it,
while the builder's function has 12 steps and the record and plan planes agree at the executed step (MEASURED, the same
one-step function at the head and on the unit tree; 11_wave2/resid/report.md, Fix Round). ``planes(...,
candidate_refusal=True)`` records such a refusal (``Planes.candidate_refusal``, the candidate then None) instead of
raising it; by default it raises, as before, and any other failure raises either way."""

from __future__ import annotations

import contextlib
import io

import pytest

from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

from . import certified as CF
from . import templates as TV

PLAN = {"step": "floor"}


@pytest.fixture(scope="module")
def program():
    vehicle = TV.template_vehicle("t0_53")
    platform = TV.residency_platform(vehicle, 2.0, PLAN)
    platform["weight_bits"] = int(vehicle.config.get("weight_bits", 8))
    with contextlib.redirect_stdout(io.StringIO()):
        hybrid = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    return vehicle, platform, hybrid, CF.Walk(tuple(vehicle.specs), vehicle.census, vehicle.program)


def _planes(program, **kwargs):
    vehicle, platform, hybrid, walk = program
    return CF.planes(hybrid, vehicle.ir, walk, platform, PLAN, timesteps=int(vehicle.config["simulation_steps"]),
                     semantics=StageSemantics(), **kwargs)


def _refusing(monkeypatch):
    def refuse(**_kwargs):
        raise ResidencyPlanError("residency_plan index 10 is outside the step function (1 steps, indices 0..0)")

    monkeypatch.setattr(CF, "collect_noc_fragments", refuse)


class TestTheCandidatePlanesRefusal:
    def test_where_the_candidate_plane_composes_the_option_changes_nothing(self, program):
        a, b = _planes(program), _planes(program, candidate_refusal=True)
        assert a.candidate_refusal is None and b.candidate_refusal is None
        for x, y in ((a.record, b.record), (a.plan, b.plan), (a.candidate, b.candidate)):
            assert CF.pass_rows(x) == CF.pass_rows(y) and CF.carried(x) == CF.carried(y)
        assert len(b.record.passes) >= 2 and CF.differences(b.record, b.plan) == [] == CF.differences(b.record,
                                                                                                     b.candidate)

    def test_by_default_the_candidate_planes_refusal_is_raised(self, program, monkeypatch):
        _refusing(monkeypatch)
        with pytest.raises(ResidencyPlanError, match="outside the step function"):
            _planes(program)

    def test_with_the_option_the_refusal_is_recorded_beside_the_record_and_plan_planes(self, program, monkeypatch):
        want = _planes(program)
        _refusing(monkeypatch)
        got = _planes(program, candidate_refusal=True)
        assert got.candidate is None
        assert got.candidate_refusal == ("ResidencyPlanError: residency_plan index 10 is outside the step function "
                                         "(1 steps, indices 0..0)")
        assert CF.pass_rows(got.record) == CF.pass_rows(want.record) and CF.pass_rows(got.plan) == CF.pass_rows(want.plan)
        assert CF.differences(got.record, got.plan) == [] and got.composition.step == want.composition.step

    def test_a_failure_that_is_no_refusal_is_raised_with_the_option(self, program, monkeypatch):
        def broken(**_kwargs):
            raise KeyError("not a refusal by name")

        monkeypatch.setattr(CF, "collect_noc_fragments", broken)
        with pytest.raises(KeyError):
            _planes(program, candidate_refusal=True)
