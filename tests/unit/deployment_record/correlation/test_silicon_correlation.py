"""The acceptance contract: every shipped target reproduces its own published numbers.

This is the suite that decides whether a physics profile may ship. A profile whose
constants cannot reproduce the measurements they were sourced from is not a
characterization, it is a guess with citations.
"""

import pytest

from mimarsinan.deployment_record.correlation.library import (
    available_cases,
    correlate_all,
    get_case,
)
from mimarsinan.deployment_record.correlation.run import correlate

ACCEPTED_PCT = 25.0


@pytest.mark.parametrize("name", sorted(available_cases()))
def test_every_shipped_case_reproduces_its_published_value(name):
    result = correlate(get_case(name))
    for axis in result.axes:
        assert axis.predicted is not None, (
            f"{name}/{axis.name}: the physics refused the axis — {axis.refusal}")
        assert abs(axis.error_pct) <= ACCEPTED_PCT, (
            f"{name}/{axis.name}: predicted {axis.predicted:.6g} vs published "
            f"{axis.published:.6g} = {axis.error_pct:+.1f}%, outside "
            f"+/-{ACCEPTED_PCT}%")


def test_the_suite_covers_at_least_three_distinct_devices():
    """One chip reproducing itself is calibration; three is correlation."""
    profiles = {get_case(n).profile for n in available_cases()}
    assert len({p for p in profiles if p != "generic_estimated_22nm"}) >= 3, profiles


def test_every_device_carries_at_least_one_independent_case():
    """A device validated only against the point its constants came from has not
    been validated at all."""
    independent = {}
    for name in available_cases():
        case = get_case(name)
        independent.setdefault(case.profile, False)
        independent[case.profile] |= case.is_independent
    assert all(independent.values()), independent


def test_every_device_discloses_the_basis_of_its_constants():
    """The basis is a property of the CONSTANTS, not of the measurement they are
    checked against: Loihi's per-op energies are pre-silicon (Davies' Table 2 says
    so) even though Frady measures a real board. A projection correlating against a
    published measurement would be a coincidence, so none may ship."""
    kinds = {get_case(n).profile: correlate(get_case(n)).measurement_kind
             for n in available_cases()}
    assert "projection" not in kinds.values(), kinds
    assert sum(1 for k in kinds.values() if k == "silicon") >= 2, kinds


def test_the_aggregate_only_model_is_what_the_suite_rejects():
    """The regression that motivated this suite: pricing TrueNorth's 96 Hz point
    with the operating-point AVERAGE (26 pJ/event) instead of the marginal energy
    overshoots by more than 3x. If a future edit reinstates the aggregate as the
    compute constant, this must fail."""
    result = correlate(get_case("truenorth_sweep_96hz"))
    axis = next(a for a in result.axes if a.name == "average_power_mw")
    neurons, rate, synapses = 4096 * 256, 95.93, 128
    aggregate_only = 26e-12 * neurons * rate * synapses * 1e3
    assert aggregate_only > 3.0 * axis.published, aggregate_only
    assert abs(axis.error_pct) <= ACCEPTED_PCT


def test_no_case_passes_by_refusing_everything():
    for result in correlate_all():
        assert result.axes, f"{result.case.name} compares nothing"
