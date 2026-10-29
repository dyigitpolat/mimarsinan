"""[U-C3, owner decision B3] Write energy per cell of NeuRRAM and HERMES-64 from their primaries.

Source: the B3 fetch intake (``12_tcad_readiness/fetch_intake/constants_write_energy_b3.md``, 2026-10-28), from the five
primaries the owner fetched (NeuRRAM refs. 41 and 52, HERMES-64 refs. 18, 28 and 30) and the NeuRRAM code release.

- NeuRRAM declares the derived band per programming pulse into one RRAM device, its verify read, and keeps its
  published 8.52 pulses per device: the write law prices the commit energy where it refused before.
- HERMES-64's write energy is not determinable from its primaries: it stays refused by name, and the profile names
  the missing quantity (the programming-rail voltage, printed only as "beyond 2.5 V") and states the derived floor as a
  boundary. Its write time, iterations and endurance rows are unchanged.
"""

import math

import pytest

from mimarsinan.deployment_record.cost.absolute import price_write_terms
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profiles_dir
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

#: The B3 intake's band per pulse into one RRAM device, pJ (sec. 2.3): low = word line at 3.0 V (432 fF) plus the
#: first set pulse at gmin; high = the 4.0 V code ceiling across 40 uS plus the reset word line at 5.0 V; nominal at
#: the log-space centre.
E_WRITE = (3.888 + 1.44, math.sqrt((3.888 + 1.44) * (10.8 + 640.0)), 10.8 + 640.0)
#: The verify read, pJ: 0.2 V across 1-40 uS for 1-10 us, nominal at the log-space centre.
E_VERIFY = (0.04, 0.8, 16.0)
ITERATIONS = 8.52
#: (e_write + e_verify) x 8.52 pulses per device write, pJ: the intake's 45.7 / 508.5 / 5,681 pJ.
PER_DEVICE_WRITE = tuple((w + v) * ITERATIONS for w, v in zip(E_WRITE, E_VERIFY))
CELLS = 100


def _row(profile, key):
    return get_platform_physics(profile).constants[key]


def _description(profile):
    return (profiles_dir() / f"{profile}.md").read_text(encoding="utf-8")


def _section(text, heading):
    """The body of one ``## heading`` section of a profile description, its line breaks read as spaces."""
    body = text.split(f"## {heading}\n", 1)[1]
    return " ".join(body.split("\n## ", 1)[0].split())


def _census(cells=CELLS, producer=0):
    return PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                                               written_cells_backing_store=cells, written_cells_producer=producer,
                                               payload_bytes=0, compute_steps=1),))


def _pricing(profile, k=1, cells=CELLS, producer=0):
    return price_write_terms(_census(cells, producer), get_platform_physics(profile), WritePolicy(batch_k=k))


def _terms(pricing):
    return {t.name: t for t in pricing.terms}


def _refusals(pricing):
    return {r.name: r.reason for r in pricing.refusals}


class TestNeuRRAMWriteEnergyRow:
    def test_the_band_per_pulse_into_one_device(self):
        row = _row("neurram", "e_write_per_cell")
        assert (row.low, row.nominal, row.high) == pytest.approx(E_WRITE, rel=1e-12)
        assert row.unit == "pJ" and row.evidence_kind == "derived"
        assert row.nominal == pytest.approx(math.sqrt(row.low * row.high), rel=1e-12)

    def test_the_derivation_shows_the_arithmetic_of_both_corners(self):
        derivation = _row("neurram", "e_write_per_cell").derivation
        for marker in ("256 x 1.5 fF + 48 fF = 432 fF", "432 fF x 3.0^2 = 3.888 pJ", "1.2^2 x 1 uS x 1 us = 1.44 pJ",
                       "3.888 + 1.44 = 5.328 pJ", "4.0^2 x 40 uS x 1 us = 640 pJ", "432 fF x 5.0^2 = 10.8 pJ",
                       "10.8 + 640 = 650.8 pJ", "log-space centre"):
            assert marker in derivation, marker

    def test_the_sources_are_located_by_page_and_line(self):
        row = _row("neurram", "e_write_per_cell")
        assert row.citation.startswith("wan2022a Methods, PDF p.10 and p.14")
        for marker in ("10.5281/zenodo.6558399", "python/wupdate_control.py:79, :90, :371-413", "end_voltage=4.0",
                       "vset_wl 3.0", "3.3"):
            assert marker in row.citation, marker
        for quote in ("each of the 256 access transistors contributes about 1.5 fF to a WL",
                      "WL drivers combined contribute about 48 fF to each WL",
                      "we set the initial set pulse voltage to be 1.2 V",
                      "We used gmin = 1 μS for all the models, and gmax = 40 μS for CNNs"):
            assert quote in row.derivation, quote

    def test_the_note_names_the_readings_the_missing_quantity_and_the_device_count(self):
        note = _row("neurram", "e_write_per_cell").note
        # [U-C4] the device count is the bundle's (cells_per_weight 2): a written weight is priced for both devices
        for marker in ("a reading, not a bound", "one RRAM device", "selector's write current at 3.0 V (set) and 5.0 V "
                       "(reset) gate bias", "measured programming energy", "differential pair",
                       "priced for both of its devices", "91.5 / 1,017 / 11,362 pJ",
                       "1.5 V in the paper", "1.2 V (in range) or 2.5 V (zero target) in the code",
                       "30 set-reset polarity reversals in the paper", "50, 30 or 20 iterations in the code",
                       "R1", "R2", "R3", "R4"):
            assert marker in note, marker


class TestNeuRRAMVerifyRead:
    def test_the_band(self):
        row = _row("neurram", "e_verify_read_per_cell")
        assert (row.low, row.nominal, row.high) == pytest.approx(E_VERIFY, rel=1e-12)
        assert row.unit == "pJ" and row.evidence_kind == "derived"

    def test_the_derivation_and_its_sources(self):
        row = _row("neurram", "e_verify_read_per_cell")
        assert "python/adc_control.py:142" in row.citation and "wan2022a Methods, PDF p.10" in row.citation
        for marker in ("vread=1.1, vref=0.9", "0.2 V", "A RRAM read takes 1–10 μs, depending on its "
                       "conductance.", "0.2^2 x 1 uS x 1 us = 0.04 pJ", "0.2^2 x 40 uS x 10 us = 16 pJ", "0.8 pJ"):
            assert marker in row.derivation, marker
        assert "ohmic" in row.note and "ADC" in row.note


class TestNeuRRAMUnchangedRows:
    def test_the_pulses_per_device_stay_published(self):
        row = _row("neurram", "write_verify_iterations")
        assert (row.low, row.nominal, row.high) == (ITERATIONS, ITERATIONS, ITERATIONS)
        assert row.evidence_kind == "published" and row.citation == "wan2022a Methods, PDF p.10"

    def test_the_write_time_and_wear_rows_stay(self):
        time = _row("neurram", "t_write_per_unit")
        assert (time.low, time.nominal, time.high) == (2.0, 6.5728, 11.0)
        assert _row("neurram", "pulses_per_write").nominal == ITERATIONS


class TestNeuRRAMWriteLaw:
    def test_the_period_write_energy_prices_per_device_write_times_the_cells(self):
        pricing = _pricing("neurram")
        assert "energy_write_mj" not in _refusals(pricing)
        band = _terms(pricing)["energy_write_mj"].band
        expected = tuple(p * 1e-12 * CELLS * 1e3 for p in PER_DEVICE_WRITE)
        assert (band.low, band.nominal, band.high) == pytest.approx(expected, rel=1e-12)
        assert band.basis == "e_write_per_cell x runtime_written_cells x write_verify_iterations"
        without_verify = tuple(w * ITERATIONS * 1e-12 * CELLS * 1e3 for w in E_WRITE)
        assert band.high == pytest.approx(without_verify[2] + 16.0 * ITERATIONS * 1e-12 * CELLS * 1e3, rel=1e-12)

    def test_the_per_device_write_is_the_intake_s_45_7_to_5681_pj(self):
        assert PER_DEVICE_WRITE == pytest.approx((45.73536, 508.5175872019541, 5681.136), rel=1e-12)

    def test_the_batch_divides_the_period_write(self):
        one, four = (_terms(_pricing("neurram", k=k))["energy_write_mj"].band for k in (1, 4))
        assert (four.low, four.nominal, four.high) == pytest.approx((one.low / 4, one.nominal / 4, one.high / 4))

    def test_producer_cells_are_priced_per_inference(self):
        band = _terms(_pricing("neurram", k=4, cells=0, producer=CELLS))["energy_write_mj"].band
        assert band.nominal == pytest.approx(PER_DEVICE_WRITE[1] * 1e-12 * CELLS * 1e3, rel=1e-12)


class TestNeuRRAMDescription:
    def test_the_validity_note_states_the_declared_band(self):
        notes = get_platform_physics("neurram").validity.notes
        assert "the write energy is not stated and refuses by name" not in notes
        assert "write energy per pulse is a derived band" in notes

    def test_the_rows_list_both_energy_rows(self):
        rows = _section(_description("neurram"), "Rows")
        assert "| `e_write_per_cell` | 5.33 / 58.9 / 650.8 pJ" in rows
        assert "| `e_verify_read_per_cell` | 0.04 / 0.8 / 16 pJ" in rows

    def test_the_absent_rows_no_longer_list_the_write_energy(self):
        absent = _section(_description("neurram"), "Absent, refused by name")
        assert not absent.lstrip().startswith("Write energy")
        assert "12.3 pJ" not in absent


class TestHermesWriteEnergyStaysRefused:
    def test_the_profile_declares_no_write_or_verify_energy(self):
        assert not {"e_write_per_cell", "e_verify_read_per_cell"} & set(get_platform_physics("hermes_64").constants)

    def test_the_write_law_refuses_it_by_name(self):
        assert "e_write_per_cell" in _refusals(_pricing("hermes_64"))["energy_write_mj"]

    def test_the_profile_names_the_missing_quantity(self):
        notes = get_platform_physics("hermes_64").validity.notes
        for marker in ("programming-rail voltage", "'beyond 2.5 V'", "khaddamaljameh2022hermescorea p.1028",
                       "measured energy per programming pulse"):
            assert marker in notes, marker

    def test_the_description_states_the_floor_as_a_boundary(self):
        absent = _section(_description("hermes_64"), "Absent, refused by name")
        for marker in ("programming-rail voltage", "To support the high voltages beyond 2.5 V that are required for "
                       "PCM programming", "> 218.75 pJ per RESET pulse", "> 70.3-85.9 pJ per SET pulse",
                       "> 39.06-218.75 pJ per iterative pulse", "> 945.3-960.9 pJ per unit-cell set-up",
                       "a floor, not a value", "no vocabulary row", "t_write_setup_per_event"):
            assert marker in absent, marker

    def test_the_description_names_what_does_not_transfer(self):
        absent = _section(_description("hermes_64"), "Absent, refused by name")
        for marker in ("khaddamaljameh2021a", "90 nm", "between 1.125 V and 1.275 V", "100 ns",
                       "at most 10 iterative steps", "papandreou2011programming", "mean"):
            assert marker in absent, marker


class TestHermesUnchangedRows:
    def test_the_iterations(self):
        row = _row("hermes_64", "write_verify_iterations")
        assert (row.low, row.nominal, row.high) == (1.0, 5.4772, 30.0)

    def test_the_write_time(self):
        step, setup = _row("hermes_64", "t_write_per_unit"), _row("hermes_64", "t_write_setup_per_event")
        assert (step.low, step.nominal, step.high) == (0.452608, 0.649216, 1.042432)
        assert (setup.low, setup.nominal, setup.high) == (1.536, 1.536, 1.6384)

    def test_the_lifetime_still_prices(self):
        pricing = price_write_terms(_census(), get_platform_physics("hermes_64"), WritePolicy(batch_k=1),
                                    most_written=1)
        assert "lifetime_inferences" in _terms(pricing)
