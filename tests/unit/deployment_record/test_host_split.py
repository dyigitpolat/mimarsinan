"""CL0: every energy verdict read twice, with and without the host's energy; a verdict that flips is host-bound."""

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.host_split import (
    HOST_BOUND,
    DualVerdict,
    HostEnergySplit,
    band_order,
    cap_verdict,
    corner_order,
    dual_verdict,
    host_energy_split,
    lowest,
)
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.schema.provenance import Band

from unit.deployment_record.g3_fixtures import quantities


def _physics(p_host=(5.0, 40.0, 280.0)):
    """A chip priced by one per-event aggregate and a host block: the headline is compute + host."""
    low, nominal, high = p_host
    return profile_from_dict({
        "format_version": 1, "name": "t", "display_name": "T", "description_file": "t.md",
        "validity": {"measurement_kind": "mixed"},
        "constants": {
            "e_synaptic_event_total": {"low": 5.0, "nominal": 10.0, "high": 20.0, "unit": "pJ",
                                       "evidence_kind": "estimated", "note": "n"},
            "p_host": {"low": low, "nominal": nominal, "high": high, "unit": "W",
                       "evidence_kind": "estimated", "note": "n"},
            "host_macs_per_s": {"nominal": 10.0, "unit": "G/s", "evidence_kind": "measured", "note": "n"},
            "t_host_op_overhead": {"low": 10.0, "nominal": 20.0, "high": 40.0, "unit": "us",
                                   "evidence_kind": "measured", "note": "n"},
        },
    })


def _priced(events=1e8, host_macs=1e6, ops=10):
    return price_absolute(quantities(synaptic_events=events, host_macs=host_macs, compute_op_count=ops),
                          _physics())


def _term(name, low, nominal, high):
    return CostTerm(name=name, unit="mJ", value=nominal, band=Band(low, nominal, high, "b"),
                    kind="modeled", source="s")


def _split(low, nominal, high, host=None):
    return HostEnergySplit(with_host=Band(low, nominal, high, "w"),
                           without_host=Band(low, nominal, high, "c") if host is None else Band(
                               low - host[0], nominal - host[1], high - host[2], "c"),
                           host=None if host is None else Band(*host, "h"))


class TestTheSplit:
    def test_without_host_is_the_headline_less_the_host_term_at_every_end(self):
        pricing = _priced()
        terms = {t.name: t for t in pricing.terms}
        split = host_energy_split(pricing.terms)
        head, host = terms["energy_per_inference_mj"].band, terms["energy_host_mj"].band
        assert split.with_host == head
        for end in ("low", "nominal", "high"):
            assert getattr(split.without_host, end) == pytest.approx(
                getattr(head, end) - getattr(host, end), rel=1e-12)
            assert getattr(split.host, end) == getattr(host, end)

    def test_without_host_equals_the_chip_energy(self):
        """Here the chip energy is the compute term alone: events x e_synaptic_event_total, end by end."""
        split = host_energy_split(_priced(events=1e8).terms)
        assert split.without_host.low == pytest.approx(1e8 * 5e-12 * 1e3, rel=1e-12)
        assert split.without_host.high == pytest.approx(1e8 * 20e-12 * 1e3, rel=1e-12)

    def test_the_host_share_is_read_at_each_end(self):
        split = host_energy_split(_priced().terms)
        share = split.host_share()
        for end, value in zip(("low", "nominal", "high"), share):
            assert value == pytest.approx(getattr(split.host, end) / getattr(split.with_host, end), rel=1e-12)

    def test_a_program_without_host_work_has_one_energy(self):
        split = host_energy_split(_priced(host_macs=0, ops=0).terms)
        assert split.host is None
        assert split.without_host == split.with_host
        assert split.host_share() == (0.0, 0.0, 0.0)

    def test_a_refused_headline_has_no_split(self):
        pricing = price_absolute(quantities(host_macs=0, compute_op_count=0), _physics())
        assert "energy_per_inference_mj" not in {t.name for t in pricing.terms}
        assert host_energy_split(pricing.terms) is None

    def test_a_host_term_without_its_headline_has_no_split(self):
        assert host_energy_split([_term("energy_host_mj", 1.0, 2.0, 3.0)]) is None

    def test_rounding_never_unorders_the_chip_band(self):
        """A host share near 1 leaves the chip energy as a small difference; the band stays ordered."""
        # Exact chip energy 0.6515929727227628 at both ends; the float headline less the host unorders it.
        terms = [_term("energy_per_inference_mj", 1.4403163238582761, 1.4404101834450502, 1.5),
                 _term("energy_host_mj", 0.7887233511355132, 0.7888172107222874, 0.8)]
        assert (1.4403163238582761 - 0.7887233511355132) > (1.4404101834450502 - 0.7888172107222874)
        split = host_energy_split(terms)
        assert split.without_host.low <= split.without_host.nominal <= split.without_host.high


class TestTheStockVerdicts:
    def test_corner_order_compares_the_same_end_of_both_bands(self):
        assert corner_order(Band(1, 2, 3, "a"), Band(2, 3, 4, "b")) == "below"
        assert corner_order(Band(2, 3, 4, "a"), Band(1, 2, 3, "b")) == "above"
        assert corner_order(Band(1, 3, 5, "a"), Band(2, 3, 4, "b")) == "straddles"
        assert corner_order(Band(1, 2, 3, "a"), Band(1, 2, 3, "b")) == "equal"

    def test_band_order_is_the_interval_rule(self):
        assert band_order(Band(1, 2, 3, "a"), Band(4, 5, 6, "b")) == "below"
        assert band_order(Band(4, 5, 6, "a"), Band(1, 2, 3, "b")) == "above"
        assert band_order(Band(1, 2, 3, "a"), Band(2, 3, 4, "b")) == "overlaps"

    def test_lowest_names_the_point_below_every_other(self):
        bands = {"a": Band(1, 2, 3, "a"), "b": Band(2, 3, 4, "b"), "c": Band(3, 4, 5, "c")}
        assert lowest(bands) == "a"
        assert lowest(bands, order=band_order) is None
        assert lowest({"a": Band(1, 3, 5, "a"), "b": Band(2, 3, 4, "b")}) is None

    def test_cap_verdict_reads_every_end(self):
        assert cap_verdict(Band(1, 2, 3, "a"), 3.5) == "admitted"
        assert cap_verdict(Band(1, 2, 3, "a"), 0.5) == "refused"
        assert cap_verdict(Band(1, 2, 3, "a"), 2.5) == "straddles"


class TestTheDualVerdict:
    def test_a_shared_host_term_never_flips_a_corner_order(self):
        """One program on two chips carries one host energy: added at the same end of both bands, it
        cancels in every corner comparison, so the verdict cannot be host-bound."""
        host = (5.0, 40.0, 280.0)
        splits = {"a": _split(6.0, 41.0, 281.0, host), "b": _split(7.0, 42.0, 282.0, host)}
        verdict = dual_verdict(lambda e: corner_order(e["a"], e["b"]), splits)
        assert verdict.with_host == verdict.without_host == "below"
        assert not verdict.host_bound and verdict.label == ""

    def test_different_host_work_flips_and_is_labelled_host_bound(self):
        """The chip-cheaper program carries more host work: below without the host, above with it."""
        splits = {"on_chip": _split(10.0, 11.0, 12.0),
                  "host_heavy": _split(20.0, 30.0, 40.0, host=(18.0, 27.5, 36.0))}
        verdict = dual_verdict(lambda e: corner_order(e["host_heavy"], e["on_chip"]), splits)
        assert verdict.without_host == "below" and verdict.with_host == "above"
        assert verdict.host_bound and verdict.label == HOST_BOUND

    def test_the_line_prints_both_verdicts_and_the_label(self):
        line = DualVerdict(with_host="above", without_host="below").line("energy: a vs b")
        assert line == "energy: a vs b: with host above; without host below; host-bound"
        assert DualVerdict(with_host="below", without_host="below").line("x") == (
            "x: with host below; without host below")

    def test_a_cap_admission_flips_where_the_host_crosses_the_cap(self):
        splits = {"p": _split(2.0, 3.0, 4.0, host=(1.0, 1.5, 2.0))}
        verdict = dual_verdict(lambda e: cap_verdict(e["p"], 2.5), splits)
        assert verdict.without_host == "admitted" and verdict.with_host == "straddles"
        assert verdict.host_bound

    def test_priced_programs_carry_through(self):
        """End to end on priced terms: a dispatch-heavy program flips against an on-chip one."""
        on_chip = host_energy_split(_priced(events=2e8, host_macs=0, ops=0).terms)
        host_heavy = host_energy_split(_priced(events=1e8, host_macs=0, ops=1000).terms)
        verdict = dual_verdict(lambda e: corner_order(e["h"], e["c"]), {"h": host_heavy, "c": on_chip})
        assert verdict.without_host == "below" and verdict.with_host == "above"
        assert verdict.label == HOST_BOUND
