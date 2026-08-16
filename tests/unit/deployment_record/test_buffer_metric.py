"""B — the required pass buffer is a mapping performance metric of the sealed
program: record-side axes over the pass-carry census, plus an OPTIONAL
declared platform ceiling (``pass_buffer_capacity_bytes``) enforced at
mapping time, BEFORE any simulation runs. Never a search axis — a candidate
has no pass structure to size.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.pass_carry import (
    carry_record_from_mapping,
    enforce_declared_buffer_capacity,
)
from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM

from unit.models.test_hybrid_pass_carry import _deep_lif_ir, _fused, _scheduled


class TestTheDeclaredCapacityGate:
    def _carrying(self):
        return _scheduled(_deep_lif_ir(), count=2)

    def test_a_program_over_the_declared_ceiling_refuses_naming_both(self):
        mapping = self._carrying()
        census = carry_record_from_mapping(mapping, 32, VERBATIM)
        assert census is not None
        with pytest.raises(ValueError) as excinfo:
            enforce_declared_buffer_capacity(
                mapping, 32, VERBATIM, census.peak_live_bytes - 1,
            )
        message = str(excinfo.value)
        assert str(census.peak_live_bytes) in message
        assert str(census.peak_live_bytes - 1) in message
        assert "pass_buffer_capacity_bytes" in message

    def test_a_program_at_the_ceiling_passes(self):
        mapping = self._carrying()
        census = carry_record_from_mapping(mapping, 32, VERBATIM)
        assert census is not None
        enforce_declared_buffer_capacity(
            mapping, 32, VERBATIM, census.peak_live_bytes,
        )

    def test_zero_means_undeclared_and_gates_nothing(self):
        enforce_declared_buffer_capacity(self._carrying(), 32, VERBATIM, 0)
        enforce_declared_buffer_capacity(self._carrying(), 32, VERBATIM, None)

    def test_a_carry_free_program_never_gates(self):
        """No pass boundary, no buffer requirement — a declared ceiling on a
        fused program is simply idle."""
        enforce_declared_buffer_capacity(_fused(_deep_lif_ir()), 32, VERBATIM, 1)


class TestTheBufferAxes:
    def test_both_axes_are_registered_after_the_traffic_axis(self):
        keys = [spec.key for spec in OBJECTIVES.all()]
        hops = keys.index("noc_total_hops")
        assert keys.index("carry_peak_live_bytes") > hops
        assert keys.index("carried_raster_bytes") > hops

    def test_the_axes_are_record_only(self):
        """A candidate has no pass structure to size, so no search mode may
        offer the buffer axes — they are mapping performance metrics."""
        assert OBJECTIVES.modes_available("carry_peak_live_bytes") == ()
        assert OBJECTIVES.modes_available("carried_raster_bytes") == ()

    def test_a_sealed_carrying_record_answers_both(self):
        from dataclasses import replace

        from mimarsinan.deployment_record.objectives.views import (
            DeploymentRecordView,
        )
        from mimarsinan.deployment_record.schema import PassCarryRecord

        from unit.deployment_record.record_fixtures import make_full_record

        record = make_full_record()
        carry = PassCarryRecord(
            transfer=VERBATIM, carried_wires=3, carried_bytes=24,
            peak_live_bytes=12, timesteps=32,
        )
        record = replace(record, schedule=replace(record.schedule, carry=carry))
        view = DeploymentRecordView(record=record)
        peak = OBJECTIVES.get("carry_peak_live_bytes")
        carried = OBJECTIVES.get("carried_raster_bytes")
        assert peak.available(view) and peak.value(view) == 12.0
        assert carried.available(view) and carried.value(view) == 24.0

    def test_a_carry_free_record_answers_neither(self):
        from mimarsinan.deployment_record.objectives.views import (
            DeploymentRecordView,
        )

        from unit.deployment_record.record_fixtures import make_full_record

        view = DeploymentRecordView(record=make_full_record())
        assert not OBJECTIVES.get("carry_peak_live_bytes").available(view)
        assert not OBJECTIVES.get("carried_raster_bytes").available(view)
