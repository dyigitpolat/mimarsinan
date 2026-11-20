"""[U-ENERGY class 3] Core-local memory access energy, read and write apart: two vocabulary rows over two
censuses on both planes (the used rows read into the arrays, the used columns written out, per evaluation step, at the
value width or one bit per step on a spiking core), priced as row x census, named where a row is undeclared and the
work exists. Every constant here is a test constant."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

ROWS = (("e_local_mem_read_per_byte", "local_mem_read_bytes", "local_memory_read"),
        ("e_local_mem_write_per_byte", "local_mem_write_bytes", "local_memory_write"))


def _physics(constants):
    return profile_from_dict({
        "format_version": 1, "name": "local", "display_name": "local", "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()},
    })


def _q(**extra):
    base = {"synaptic_events": 1e6, "cores_physical": 4.0, "latency_steps": 10.0, "host_macs": 0.0,
            "compute_op_count": 0.0, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in base.items()})


def _term(pricing, name):
    return next(t for t in pricing.terms if t.name == name)


def _candidate(**context):
    from mimarsinan.deployment_record.quantities.from_candidate import CandidateQuantityContext, from_candidate
    return from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                          context=CandidateQuantityContext(**context))


class TestTheRows:
    @pytest.mark.parametrize("row,census,_label", ROWS)
    def test_each_row_is_an_energy_over_its_census(self, row, census, _label):
        spec = spec_for(row)
        assert (spec.multiplicand, spec.display_unit) == (census, "pJ")

    def test_priced_is_each_row_times_its_census(self):
        pricing = price_absolute(_q(local_mem_read_bytes=5000.0, local_mem_write_bytes=700.0),
                                 _physics({"e_mac": 0.5, "t_cycle": 100.0, "e_local_mem_read_per_byte": 0.6,
                                           "e_local_mem_write_per_byte": 1.1}))
        assert _term(pricing, "energy_dynamic_mj").value == pytest.approx(
            (0.5e-12 * 1e6 + 0.6e-12 * 5000 + 1.1e-12 * 700) * 1e3, rel=1e-12)

    @pytest.mark.parametrize("row,census,label", ROWS)
    def test_named_unpriced_where_the_row_is_undeclared_and_the_work_exists(self, row, census, label):
        pricing = price_absolute(_q(**{census: 5000.0}), _physics({"e_mac": 0.5, "t_cycle": 100.0}))
        assert f"{row}({label})" in _term(pricing, "energy_dynamic_mj").source

    def test_the_generic_projection_declares_estimated_bands_and_a_write_no_cheaper_than_a_read(self):
        physics = get_platform_physics("generic_estimated_22nm")
        read, write = (physics.constants[row] for row, _c, _l in ROWS)
        for constant in (read, write):
            assert constant.evidence_kind == "estimated" and constant.low < constant.nominal < constant.high
        for corner in ("low", "nominal", "high"):
            assert getattr(physics.band(ROWS[1][0]), corner) >= getattr(physics.band(ROWS[0][0]), corner)


class TestTheCensusOnTheCandidatePlane:
    def test_value_domain_bytes_are_elements_at_the_value_width(self):
        q = _candidate(local_mem_read_elements=30, output_neurons=12, value_bits=4, value_rule=object())
        assert q.get("local_mem_read_bytes").value == 15.0 and q.get("local_mem_write_bytes").value == 6.0
        assert q.get("neurons_used").value == 12.0

    def test_without_a_value_width_both_censuses_are_withheld_by_name(self):
        q = _candidate(local_mem_read_elements=30, output_neurons=12, value_bits=None, value_rule=object())
        for _row, census, _label in ROWS:
            assert "activation_bits undeclared" in q.withheld_reason(census)

    def test_a_spiking_core_moves_one_bit_per_element_per_time_step(self):
        q = _candidate(local_mem_read_elements=30, output_neurons=12, timesteps=4)
        assert q.get("local_mem_read_bytes").value == 30 * 4 / 8.0
        assert q.get("local_mem_write_bytes").value == 12 * 4 / 8.0


class TestTheCensusOnTheRecordPlane:
    def test_a_spiking_record_counts_bytes_one_bit_per_element_per_step(self):
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from unit.deployment_record.record_fixtures import make_full_record
        q = from_record(make_full_record())
        steps = q.get("timesteps").value
        assert steps > 1
        assert q.get("local_mem_read_bytes").value == pytest.approx(q.get("axons_used").value * steps / 8.0)
        assert q.get("local_mem_write_bytes").value == pytest.approx(q.get("neurons_used").value * steps / 8.0)


class TestTheTwoPlanesAgree:
    def test_the_candidates_counts_are_the_sealed_records_on_every_lane_program(self):
        """The used rows and columns the candidate's census counts over its evaluated cores equal the sealed record's
        (``axons_used``, ``neurons_used``, dark cores out) on every lane program, pipelined ones included: a pass
        restricted to the cores it places keeps each core's own used extents."""
        from mimarsinan.deployment_record.build.program_census import candidate_programming_census
        from mimarsinan.deployment_record.design_point.pricing import executed_record
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
        from unit.mapping.lanes.lane_planes import STEPS, TRANSFER
        from unit.mapping.pipelined_multiplexing.test_planes import CASES, _lane_program
        for fixture, geometry, pipelined in CASES:
            platform, program, scheduled = _lane_program(fixture, geometry, pipelined)
            census = candidate_programming_census(scheduled.noc, weight_bits=4)
            sealed = from_record(executed_record(build_hybrid_mapping_for_pipeline(program.ir, platform), platform,
                                                 weight_bits=4, label="lane program", timesteps=STEPS,
                                                 pass_transfer=TRANSFER))
            case = (fixture, geometry, pipelined)
            assert census.local_mem_read_elements == sealed.get("axons_used").value, case
            assert census.output_neurons == sealed.get("neurons_used").value, case
