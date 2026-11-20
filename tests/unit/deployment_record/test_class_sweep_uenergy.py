"""[U-ENERGY, the classes swept by the build's own search] Three sites of the unit's classes outside the design's sweep.

- S-1 (class 2, the one load law): an undeclared per-core program row over cores a census programs is named in the
  write energy's text, as the release version's per-load term named it ("e_core_program(core_program)"), never
  dropped silently.
- S-2 (class 2, the one static sum): a declared carry-buffer static row over a capacity the view does not produce is
  named with the plane's reason, and over a capacity its platform's declarations make zero prints that zero's reason
  in the static leg's basis.
- S-3 (classes 1 and 2, the level plane): the capability level's plane states its passes, so it hands the one load
  law its two load censuses (never refused as a view without passes), and its view puts every absent factor in one
  of the four states with its reason, as the candidate and record planes do.

Every constant is a test constant or a shipped description as it ships."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.walls.loads import NO_LOAD_CENSUS
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.load_census_fixtures import written_program
from unit.deployment_record.printed_names import named
from unit.deployment_record.registered_hosts import registered_fixture
from unit.deployment_record.test_level_pricing import _HOST_NAME, _HOST_ROWS

BUFFER_STATIC = "p_carry_buffer_static_per_byte"
#: [composition: U-HOST] the level pricing's physics declares its host rows as a registered test description's own.
_level_host = registered_fixture(_HOST_NAME, _HOST_ROWS, autouse=False)


def _physics(constants, name="sweep"):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()},
    })


def _q(**values):
    base = {"synaptic_events": 1e6, "cores_physical": 4.0, "latency_steps": 10.0, "host_macs": 0.0,
            "compute_op_count": 0.0, "timesteps": 1.0, **values}
    return {k: QuantityValue(float(v), "static") for k, v in base.items()}


class TestS1AnUndeclaredPerCoreProgramRowIsNamed:
    BYTES = {"e_mac": 0.5, "t_cycle": 100.0, "e_program_per_byte": 3.0, "t_program_per_byte": 7.0}

    def test_the_load_names_it_over_the_cores_it_programs(self):
        pricing = price_absolute(Quantities(_q()), _physics(self.BYTES), loads=written_program((250, 250), (250,)))
        terms = {t.name: t for t in pricing.terms}
        for name in ("energy_load_mj", "energy_programming_mj"):
            assert "e_core_program(core_program)" in terms[name].source, name

    def test_the_writes_at_a_declared_batch_name_it_too(self):
        loads = written_program((250, 250), (250,))
        period = loads.period.writes
        from dataclasses import replace
        writes = WriteInputs.of(replace(period, policy=replace(period.policy, batch_k=4)))
        energy = next(t for t in price_write_inputs(writes, _physics(self.BYTES)).terms if t.name == "energy_write_mj")
        assert "unpriced: e_core_program(core_program)" in energy.source

    def test_a_declared_row_is_priced_and_a_census_that_programs_no_core_names_nothing(self):
        declared = price_absolute(Quantities(_q()), _physics({**self.BYTES, "e_core_program": 5.0}),
                                  loads=written_program((250, 250), (250,)))
        assert "e_core_program(" not in {t.name: t for t in declared.terms}["energy_load_mj"].source
        lone = price_absolute(Quantities(_q()), _physics(self.BYTES), loads=written_program((0,), (0,)))
        assert "e_core_program(" not in {t.name: t for t in lone.terms}["energy_programming_mj"].source


class TestS2TheBuffersStaticRowIsNeverASilentZero:
    STATIC = {"e_mac": 0.5, "t_cycle": 100.0, "p_static_global": 4.0, BUFFER_STATIC: 50.0}

    def _energy(self, quantities):
        terms = {t.name: t for t in price_absolute(quantities, _physics(self.STATIC)).terms}
        return terms["energy_per_inference_mj"].source, terms["energy_static_mj"].source

    def test_a_capacity_the_view_does_not_produce_is_named_with_the_reason(self):
        source, _static = self._energy(Quantities(_q()))
        assert named(source, BUFFER_STATIC) == [
            f"{BUFFER_STATIC}(static; not produced on this plane: the view states no carry_buffer_capacity_bytes)"]

    def test_a_capacity_zero_by_declaration_prints_its_reason_in_the_legs_basis(self):
        reason = "the chip declares no on-chip carry buffer"
        source, static = self._energy(Quantities(_q(), known_zero={"carry_buffer_capacity_bytes": reason}))
        assert f"zero by declaration: {BUFFER_STATIC} ({reason})" in static
        assert not named(source, BUFFER_STATIC)  # never named unpriced, never priced

    def test_a_declared_capacity_is_priced_as_before(self):
        source, static = self._energy(Quantities(_q(carry_buffer_capacity_bytes=1000.0)))
        assert "zero by declaration" not in static and not named(source, BUFFER_STATIC)
        assert BUFFER_STATIC in static


def _level_view(**kw):
    from unit.deployment_record.test_level_pricing import _view
    from unit.mapping.capability.ir_fixtures import lane_graph
    return _view(lane_graph(), "C0", kw.pop("budget", 1), **kw)


@pytest.mark.usefixtures(f"registered_host_{_HOST_NAME}")  # [composition: U-HOST]
class TestS3TheLevelPlane:
    def test_its_loads_are_priced_by_the_one_law_over_its_own_census(self):
        from mimarsinan.deployment_record.cost.absolute.level_pricing import price_level
        from unit.deployment_record.test_level_pricing import _physics as level_physics
        pricing = price_level(_level_view(), level_physics())
        terms = {t.name: t for t in pricing.terms}
        refusals = {r.name: r.reason for r in pricing.refusals}
        assert NO_LOAD_CENSUS not in refusals.values()
        assert {"latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"} <= set(terms)

    def test_its_first_load_writes_every_tile_once(self):
        """F = 4 tiles on 1 core: the first load writes the four tiles' 56 cells; the period rewrites them too."""
        view = _level_view()
        first = view.loads.first.writes.census.census.passes
        assert sum(p.written_cells_backing_store for p in first) == 56
        assert sum(p.cores - p.resident_cores for p in first) == 4

    def test_every_absent_factor_is_in_one_state_with_its_reason(self):
        from unit.deployment_record.test_no_silent_zero_uenergy import _check_view
        from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
        for view in (_level_view(budget=4), _level_view()):
            assert view.quantities.unproduced, "the level plane states what it does not produce"
            zeros = declared_zeros(False, False, True)
            assert _check_view(view.quantities, zeros) > 0

    def test_a_declared_carry_buffer_is_the_levels_chip_fact(self):
        """The level's chip states its declared carry buffer's size, which its static sum and area read; a chip that
        declares none makes it a zero by declaration."""
        from unit.deployment_record.test_level_pricing import _platform
        buffered = _level_view(platform={**_platform(inference_batch_K=16), "carry_buffer_bytes": 4096})
        assert buffered.quantities.get("carry_buffer_capacity_bytes").value == 4096
        assert _level_view().quantities.known_zero_reason("carry_buffer_capacity_bytes") == (
            "the chip declares no on-chip carry buffer")

    def test_a_value_domain_level_sends_no_spike_packets(self):
        view = _level_view(budget=4)
        for key in ("noc_total_packets", "noc_intra_tile_packets", "noc_inter_tile_packets", "noc_total_hops"):
            assert view.quantities.get(key).value == 0.0, key
