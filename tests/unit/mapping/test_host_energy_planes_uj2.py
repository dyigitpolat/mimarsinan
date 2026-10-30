"""[U-J2 K-1] Both planes compose the host's energy alike: the candidate plane (the scaling ladder's priced program and
its write-law columns at K) and the record plane (a sealed record's view, which states no segment census).

On each: the headline is the chip plus the attributable host (``energy_host_mj``: the host's work above its idle
floor, the link and host memory per crossing byte). [targeted M1] The system line ``energy_system_mj`` is the chip plus
the whole host at its corners (``energy_host_system_mj``), at every end; at the nominal (one setup) the whole host is
the attributable host plus the idle floor under the work and the idle draw over the wait, and at every end the headline
lies at or below the system line (the work's band ends are the cost ends of p_host - p_host_idle). Vehicle: U-J's
census CNN on a study chip with the edge host (``test_host_boundary_census_uj``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from unit.deployment_record.test_host_energy_attribution_uj2 import EX_D, _edge, _priced as _priced_record
from unit.mapping.test_host_boundary_census_uj import LD, _network, _platform, _priced

CORNERS = ("low", "nominal", "high")
ATTRIBUTABLE = ("energy_host_work_mj", "energy_host_link_mj", "energy_host_mem_mj")
SYSTEM = ("energy_host_floor_mj", "energy_host_idle_mj")


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))


def _end(band, corner):
    return getattr(band, corner)


def _sum(terms, names, corner):
    return sum(_end(terms[name], corner) for name in names if name in terms)


def _check_composition(terms):
    """The composition's identities; ``terms``: name -> Band. At every corner: the attributable host is its three
    terms, the headline the chip plus it, the system line the chip plus the whole host, the headline at or below the
    system line; at the nominal (one setup) the whole host is the attributable host plus the floor and the idle."""
    for corner in CORNERS:
        chip = _sum(terms, ("energy_dynamic_mj", "energy_static_mj", "energy_write_mj"), corner)
        assert _end(terms["energy_host_mj"], corner) == pytest.approx(_sum(terms, ATTRIBUTABLE, corner), rel=1e-12)
        assert _end(terms["energy_per_inference_mj"], corner) == pytest.approx(
            chip + _end(terms["energy_host_mj"], corner), rel=1e-12), corner
        assert _end(terms["energy_system_mj"], corner) == pytest.approx(
            chip + _end(terms["energy_host_system_mj"], corner), rel=1e-12), corner
        assert _end(terms["energy_per_inference_mj"], corner) <= _end(terms["energy_system_mj"], corner), corner
    assert _end(terms["energy_host_system_mj"], "nominal") == pytest.approx(
        _end(terms["energy_host_mj"], "nominal") + _sum(terms, SYSTEM, "nominal"), rel=1e-12)


class TestBothPlanesComposeAlike:
    def test_the_candidate_plane(self):
        _, priced = _priced(True)
        terms = priced.terms["program"]
        assert set(ATTRIBUTABLE) <= set(terms) and set(SYSTEM) <= set(terms)
        _check_composition(terms)

    def test_the_record_plane(self):
        terms, _ = _priced_record(_edge("study_compass"), **EX_D)
        _check_composition({name: term.band for name, term in terms.items()})

    def test_without_the_edge_rows_the_candidate_prints_no_system_line(self):
        _, priced = _priced(False)
        assert not {"energy_system_mj", "energy_host_system_mj", *ATTRIBUTABLE[:1], "energy_host_mem_mj",
                    *SYSTEM[:1]} & set(priced.terms["program"])


def test_the_write_law_columns_carry_the_attribution_at_k():
    import scaling_ladder

    network, base = _network(), _platform(True)
    census = LD.shape_census(network, base)
    scheduled = LD.schedule_program(network, census, base, scaling_ladder.program_facts,
                                    tiles=LD.packed_tiles(census, base))
    point = _platform(True, inference_batch_K=16, inference_rate_hz=30.0, min_lifetime_hours=1.0)
    columns = LD.price_scheduled(network, census, scheduled, point).write_law["terms"]
    for name in (*ATTRIBUTABLE, *SYSTEM, "energy_host_system_mj", "energy_system_mj", "energy_host_mj"):
        assert columns.get(name) is not None, name
    low, nominal, high = columns["energy_system_mj"]
    assert low <= nominal <= high
