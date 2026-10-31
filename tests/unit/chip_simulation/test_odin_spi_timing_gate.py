"""[ODIN SPI clock domain] The local gate: a routed summary's check_timing, read for the SCK domains.

``scripts/hacc/check_spi_timing.py`` reads the part of a routed Vivado timing
summary before "Timing Details" and fails on any unclocked register pin (each
root named, the SPI roots as such), any unconstrained internal endpoint, an SCK
clock count other than the one expected (the count is required whenever the
summary shows an SPI domain), an SCK domain with no timed path inside it or
across either crossing, an SCK domain whose own setup or hold cell is blank, a
crossing row into or out of an SCK clock whose setup or hold cell is blank, and
any negative slack there or in the design.

The fixtures are the archived summaries of two builds that carried no SPI
constraint (``tests/fixtures/odin_routed_timing/PROVENANCE.md``): route D
revision 5's candidate A, whose fabric-routed SCK domains failed on the card, and
Stage 1's NC 1, whose SCK sat on a global buffer but was still never timed. Both
must be red. The gate must also read the tables right, blank cells included.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "scripts" / "hacc" / "check_spi_timing.py"
FIXTURES = REPO / "tests" / "fixtures" / "odin_routed_timing"
ROUTE_D_R5_A = FIXTURES / "b71035_A_impl_1.summary.rpt"
STAGE1_NC1 = FIXTURES / "b71018_stage1_nc1.summary.rpt"

FIXTURE_SHA256 = {
    ROUTE_D_R5_A.name: "eae3335262d11cb386ba274d49b026f1f613ead9f8488855f7a234824ba624a4",
    STAGE1_NC1.name: "736f60e2cc946fd674c387e561fe258d0d1bd3172419bdb3fb7fe1e1a523133b",
}


def _gate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(GATE), *args], capture_output=True,
                          text=True)


def _module():
    spec = importlib.util.spec_from_file_location("check_spi_timing_under_test", GATE)
    assert spec is not None and spec.loader is not None, f"{GATE} is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sck_root(unit: int) -> str:
    return f"level0_i/ulp/odin_{unit}/inst/kernel_i/core_gen[0].spi_i/sck_reg/Q"


class TestTheArchivedBuildsAreRed:

    def test_the_fixtures_are_the_archived_bytes(self):
        for path in (ROUTE_D_R5_A, STAGE1_NC1):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA256[path.name]

    def test_route_d_r5_a_fails_on_every_units_spi_domain(self):
        result = _gate(str(ROUTE_D_R5_A))
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout
        assert "FAIL: 2667 register pin(s) with no clock" in out
        for unit in range(7):
            assert f"FAIL:   381 unclocked pin(s) driven by {_sck_root(unit)} (SPI SCK)" in out
        assert "FAIL: 5530 unconstrained internal endpoint(s)" in out
        assert out.rstrip().endswith("RESULT: FAIL")

    def test_stage1_nc1_fails_although_its_sck_had_a_global_buffer(self):
        result = _gate(str(STAGE1_NC1))
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout
        assert "FAIL: 381 register pin(s) with no clock" in out
        assert (f"FAIL:   381 unclocked pin(s) driven by "
                f"level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/sck_reg/Q (SPI SCK)") in out
        assert "FAIL: 790 unconstrained internal endpoint(s)" in out
        assert out.rstrip().endswith("RESULT: FAIL")

    def test_an_expected_sck_count_the_build_lacks_fails_too(self):
        result = _gate(str(STAGE1_NC1), "--expect-sck", "1")
        assert result.returncode == 1
        assert "FAIL: 0 SCK clock(s) in the clock summary, expected 1" in result.stdout

    def test_a_file_that_is_no_timing_summary_is_refused(self, tmp_path):
        other = tmp_path / "other.rpt"
        other.write_text("Design Timing Summary\nnothing here\n")
        result = _gate(str(other))
        assert result.returncode == 2
        assert "REFUSING" in result.stderr


class TestTheGateReadsTheTables:

    def test_the_check_timing_counts_and_roots(self):
        summary = _module().parse(ROUTE_D_R5_A.read_text())
        assert summary["checks"]["no_clock"] == 2667
        assert summary["checks"]["unconstrained_internal_endpoints"] == 5530
        assert summary["checks"]["no_input_delay"] == 7
        assert summary["no_clock_roots"] == [(_sck_root(u), 381) for u in range(7)]

    def test_the_design_row_and_the_kernel_clock_row(self):
        summary = _module().parse(ROUTE_D_R5_A.read_text())
        design = summary["design"]
        assert (design["WNS(ns)"], design["WHS(ns)"], design["WPWS(ns)"]) == (
            "0.003", "0.009", "0.000")
        intra = {row["Clock"]: row for row in summary["intra"]}
        kernel = intra["io_clk_freerun_00_clk_p"]
        assert (kernel["WNS(ns)"], kernel["WHS(ns)"], kernel["TNS Total Endpoints"]) == (
            "0.034", "0.010", "139268")

    def test_the_clock_tree_follows_the_indentation(self):
        clocks = _module().parse(ROUTE_D_R5_A.read_text())["clocks"]
        assert clocks["io_clk_freerun_00_clk_p"] is None
        assert clocks["clk_kernel_00_unbuffered_net"] == "io_clk_freerun_00_clk_p"
        assert clocks["hbm_aclk"] == "io_clk_freerun_00_clk_p"

    def test_a_blank_cell_reads_as_none_not_as_its_neighbour(self):
        """`hbm_aclk -> io_clk_freerun_00_clk_p` has setup paths only."""
        inter = _module().parse(ROUTE_D_R5_A.read_text())["inter"]
        row = [r for r in inter
               if (r["From Clock"], r["To Clock"]) == ("hbm_aclk", "io_clk_freerun_00_clk_p")]
        assert len(row) == 1
        assert row[0]["WNS(ns)"] == "1.594" and row[0]["TNS Total Endpoints"] == "210"
        assert row[0]["WHS(ns)"] is None and row[0]["THS Total Endpoints"] is None

    def test_the_async_groups_are_read_with_their_clocks(self):
        other = _module().parse(ROUTE_D_R5_A.read_text())["other"]
        assert other, "no Other Path Groups row read"
        assert all(row["Path Group"] == "**async_default**" for row in other)
        assert any((r["From Clock"], r["To Clock"]) == (
            "dma_ip_axi_aclk_1", "GTYE4_CHANNEL_TXOUTCLK[3]_7") and r["WNS(ns)"] == "0.457"
            for r in other)


class TestTheVerdictCoversEveryWayToHideTheDomain:
    """On parsed summaries edited in memory: each hiding path is a reason by name."""

    MASTER = "io_clk_freerun_00_clk_p"
    SCK = "level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/sck_reg/Q"

    def _timed(self, **edits):
        summary = {
            "checks": {"no_clock": 0, "unconstrained_internal_endpoints": 0},
            "no_clock_roots": [],
            "clocks": {self.MASTER: None, self.SCK: self.MASTER},
            "design": {"WNS(ns)": "0.003", "WHS(ns)": "0.009", "WPWS(ns)": "0.000"},
            "intra": [{"Clock": self.MASTER, "WNS(ns)": "0.034", "WHS(ns)": "0.010"},
                      {"Clock": self.SCK, "WNS(ns)": "15.0", "WHS(ns)": "0.030"}],
            "inter": [{"From Clock": self.MASTER, "To Clock": self.SCK,
                       "WNS(ns)": "6.0", "WHS(ns)": "18.0"},
                      {"From Clock": self.SCK, "To Clock": self.MASTER,
                       "WNS(ns)": "11.0", "WHS(ns)": "1.721"}],
            "other": [{"Path Group": "**async_default**", "From Clock": self.MASTER,
                       "To Clock": self.SCK, "WNS(ns)": "7.0", "WHS(ns)": "20.0"}],
        }
        summary.update(edits)
        return summary

    def test_a_timed_and_met_domain_passes(self):
        assert _module().verdict(self._timed(), expect_sck=1) == []

    def test_a_hold_race_inside_the_domain_fails(self):
        summary = self._timed()
        summary["intra"][1]["WHS(ns)"] = "-0.803"
        assert _module().verdict(summary, expect_sck=1) == [f"{self.SCK}: intra WHS(ns) -0.803"]

    def test_a_domain_with_no_timed_path_inside_fails(self):
        summary = self._timed(intra=self._timed()["intra"][:1])
        assert _module().verdict(summary, expect_sck=1) == [
            f"{self.SCK}: no timed path inside its domain"]

    @pytest.mark.parametrize("keep", [0, 1])
    def test_a_crossing_hidden_by_an_exception_fails(self, keep):
        inter = [self._timed()["inter"][keep]]
        fails = _module().verdict(self._timed(inter=inter), expect_sck=1)
        assert len(fails) == 1 and "no timed path across the SPI crossing" in fails[0]

    def test_a_negative_async_path_into_the_domain_fails(self):
        other = [dict(self._timed()["other"][0], **{"WHS(ns)": "-5.293"})]
        fails = _module().verdict(self._timed(other=other), expect_sck=1)
        assert fails == [f"**async_default** {self.MASTER} -> {self.SCK}: WHS(ns) -5.293"]

    def test_a_design_that_misses_timing_fails(self):
        fails = _module().verdict(self._timed(design={"WNS(ns)": "-0.111", "WHS(ns)": "0.009",
                                                      "WPWS(ns)": "0.000"}), expect_sck=1)
        assert fails == ["design WNS(ns) -0.111"]

    def test_an_unclocked_root_that_is_not_spi_is_named_as_other(self):
        summary = self._timed(checks={"no_clock": 4, "unconstrained_internal_endpoints": 0},
                              no_clock_roots=[("level0_i/ulp/x/clk_div_reg/Q", 4)])
        fails = _module().verdict(summary, expect_sck=1)
        assert fails[0] == "4 register pin(s) with no clock"
        assert fails[1] == "  4 unclocked pin(s) driven by level0_i/ulp/x/clk_div_reg/Q (other)"


STAGE1_NC1_TIMED = FIXTURES / "b71018_stage1_nc1_spi_xdc_in_memory.summary.rpt"
ROUTE_D_R5_A_TIMED = FIXTURES / "b71035_A_impl_1_spi_xdc_in_memory.summary.rpt"


class TestTheGateOnRealSummariesOfTheConstrainedDomain:
    """Vivado's own summaries with the build's XDC read, in memory, into two archived routings.

    The constraint makes the domain visible: check_timing clears in both. Where the
    SCK net sat on a global buffer (71018) the domain meets timing and the gate
    passes; where six SCK nets were fabric-routed without the constraint (71035)
    the gate names each unit's hold race, which is what a constrained build's
    router must fix.
    """

    def test_the_fixtures_are_the_probe_bytes(self):
        assert hashlib.sha256(STAGE1_NC1_TIMED.read_bytes()).hexdigest() == (
            "ac36471f80020120ad4d95f2ccb2793e66fadbcd2a830f4c8f29d5f0f06435cf")
        assert hashlib.sha256(ROUTE_D_R5_A_TIMED.read_bytes()).hexdigest() == (
            "9d7d380ce3c01a0040be77421c5a9899206a4f4e533b73dcc7817ffeab2e46f9")

    def test_a_timed_and_met_domain_passes_with_its_clock_count(self):
        result = _gate(str(STAGE1_NC1_TIMED), "--expect-sck", "1")
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert "check_timing: no_clock 0, unconstrained_internal_endpoints 0" in out
        assert f"clock {_sck_root(0)}: WNS 16.436 WHS 0.031" in out
        assert f"inter io_clk_freerun_00_clk_p -> {_sck_root(0)}: WNS 10.844 WHS 24.443" in out
        assert f"inter {_sck_root(0)} -> io_clk_freerun_00_clk_p: WNS 4.140 WHS 1.721" in out
        assert out.rstrip().endswith("RESULT: PASS")

    def test_the_same_summary_fails_when_more_masters_were_built(self):
        result = _gate(str(STAGE1_NC1_TIMED), "--expect-sck", "2")
        assert result.returncode == 1
        assert "FAIL: 1 SCK clock(s) in the clock summary, expected 2" in result.stdout

    def test_the_fabric_routed_domains_fail_by_unit_on_their_hold_races(self):
        result = _gate(str(ROUTE_D_R5_A_TIMED), "--expect-sck", "7")
        assert result.returncode == 1, result.stdout + result.stderr
        fails = [line for line in result.stdout.splitlines() if line.startswith("FAIL:")]
        races = {1: "-0.557", 2: "-0.307", 3: "-0.803", 4: "-0.363", 5: "-0.116", 6: "-0.765"}
        assert fails == [f"FAIL: {_sck_root(u)}: intra WHS(ns) {races[u]}" for u in range(1, 7)] + [
            "FAIL: design WHS(ns) -0.803"]
        assert f"clock {_sck_root(0)}: WNS 16.568 WHS 0.044" in result.stdout


def _blank_cells(text: str, table: str, names: tuple, keys: tuple) -> str:
    """The summary with the named cells of one row of ``table`` blanked, column widths kept.

    The row is the one whose leading name cells are ``names`` (the clock of an Intra Clock
    Table row; the From and To clocks of an Inter Clock Table row). That is how Vivado prints
    a check an exception removed (a false path, a clock group): the cell is empty. The cells
    are found from the table's own header and dash line, not through the gate under test.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.rstrip() == f"| {table}")
    dash = next(i for i in range(start + 1, len(lines))
                if re.match(r"^\s*-+(\s+-+)+\s*$", lines[i]))
    spans = {lines[dash - 1][a:b].strip(): (a, b)
             for a, b in ((m.start(), m.end()) for m in re.finditer(r"-+", lines[dash]))}
    end = next(i for i in range(dash + 1, len(lines)) if not lines[i].strip())
    row = next(i for i in range(dash + 1, end) if tuple(lines[i].split()[:len(names)]) == names)
    cells = list(lines[row])
    for key in keys:
        a, b = spans[key]
        token = re.search(r"\S+$", lines[row][:b])  # a number ends where its dash run ends
        assert token is not None and token.start() >= a, (key, lines[row][:b][-30:])
        cells[token.start():token.end()] = " " * (token.end() - token.start())
    lines[row] = "".join(cells)
    return "".join(lines)


def _blank_intra_cells(text: str, clock: str, keys: tuple) -> str:
    return _blank_cells(text, "Intra Clock Table", (clock,), keys)


HOLD_CELLS = ("WHS(ns)", "THS(ns)", "THS Failing Endpoints", "THS Total Endpoints")


class TestTheDomainCannotBePartlyHidden:
    """Review minor m7 (U-SP2): two ways part of the SPI domain could hide and the gate still pass.

    (a) The SCK count was checked only when ``--expect-sck`` was given, so a build that
    declared some of its SCK clocks, or none, passed on its other reasons alone. The count
    is required whenever the summary shows an SPI domain at all: an SPI root among the
    unclocked pins, or an SCK clock. A fabric with no SPI slave (a generated core) shows
    neither and needs no count.
    (b) An SCK clock whose Intra Clock Table row had no number for WHS passed, so a hold
    exception inside the domain (``set_false_path -hold``) would hide its hold check. The
    setup cell is held to the same rule.
    """

    MASTER = TestTheVerdictCoversEveryWayToHideTheDomain.MASTER
    SCK = TestTheVerdictCoversEveryWayToHideTheDomain.SCK

    def _timed(self, **edits):
        return TestTheVerdictCoversEveryWayToHideTheDomain()._timed(**edits)

    def test_a_timed_domain_without_its_expected_count_fails(self):
        result = _gate(str(STAGE1_NC1_TIMED))
        assert result.returncode == 1, result.stdout + result.stderr
        fails = [line for line in result.stdout.splitlines() if line.startswith("FAIL:")]
        assert fails == ["FAIL: SPI clock domain in the summary (1 SCK clock(s), 0 unclocked SPI "
                         "root(s)) but no expected SCK count (--expect-sck)"]
        assert result.stdout.rstrip().endswith("RESULT: FAIL")

    def test_an_untimed_domain_without_its_expected_count_is_named_too(self):
        result = _gate(str(ROUTE_D_R5_A))
        assert result.returncode == 1
        assert ("FAIL: SPI clock domain in the summary (0 SCK clock(s), 7 unclocked SPI root(s)) "
                "but no expected SCK count (--expect-sck)") in result.stdout

    def test_the_count_is_asked_for_by_the_domain_not_by_the_caller(self):
        fails = _module().verdict(self._timed())
        assert fails == ["SPI clock domain in the summary (1 SCK clock(s), 0 unclocked SPI root(s)) "
                         "but no expected SCK count (--expect-sck)"]
        assert _module().verdict(self._timed(), expect_sck=1) == []

    def test_an_spi_root_alone_asks_for_the_count(self):
        summary = self._timed(checks={"no_clock": 381, "unconstrained_internal_endpoints": 0},
                              no_clock_roots=[(self.SCK, 381)], clocks={self.MASTER: None},
                              intra=self._timed()["intra"][:1], inter=[], other=[])
        fails = _module().verdict(summary)
        assert ("SPI clock domain in the summary (0 SCK clock(s), 1 unclocked SPI root(s)) "
                "but no expected SCK count (--expect-sck)") in fails

    def test_a_fabric_with_no_spi_domain_needs_no_count(self):
        """The generated core's summaries list no SPI root and no SCK clock: they keep passing."""
        summary = self._timed(clocks={self.MASTER: None}, intra=self._timed()["intra"][:1],
                              inter=[], other=[])
        assert _module().verdict(summary) == []

    def test_an_sck_hold_check_left_blank_fails(self):
        summary = self._timed()
        summary["intra"][1]["WHS(ns)"] = None
        assert _module().verdict(summary, expect_sck=1) == [
            f"{self.SCK}: intra WHS(ns) has no number: the hold check inside the domain is not timed"]

    def test_an_sck_setup_check_left_blank_fails(self):
        summary = self._timed()
        summary["intra"][1]["WNS(ns)"] = None
        assert _module().verdict(summary, expect_sck=1) == [
            f"{self.SCK}: intra WNS(ns) has no number: the setup check inside the domain is not timed"]

    def test_a_hold_check_blanked_in_a_real_summary_fails(self, tmp_path):
        text = STAGE1_NC1_TIMED.read_text()
        blanked = _blank_intra_cells(text, _sck_root(0), (
            "WHS(ns)", "THS(ns)", "THS Failing Endpoints", "THS Total Endpoints"))
        assert len(blanked) == len(text) and blanked != text
        path = tmp_path / "hold_false_pathed.summary.rpt"
        path.write_text(blanked)
        result = _gate(str(path), "--expect-sck", "1")
        assert result.returncode == 1, result.stdout + result.stderr
        fails = [line for line in result.stdout.splitlines() if line.startswith("FAIL:")]
        assert fails == [f"FAIL: {_sck_root(0)}: intra WHS(ns) has no number: the hold check "
                         f"inside the domain is not timed"]


FIXED_CU7 = FIXTURES / "b71070_fixed_cu7_impl_1.summary.rpt"


class TestTheCrossingsCannotBeHidden:
    """Review minor n1 (U-SP2 review, closed by U-SP3): a crossing row with a blank cell.

    The Intra Clock Table row of an SCK clock must carry a number for WNS and WHS, but a
    crossing row into or out of an SCK clock failed only when it was missing or negative. A
    hold exception on a crossing (``set_false_path -hold`` from the core clock into SCK)
    blanks that row's WHS cell and hides MOSI's hold check, and the gate still passed. Every
    Inter Clock Table row into or out of an SCK clock is now held to the intra rows' rule:
    a number for WNS and for WHS, and no negative one.
    """

    MASTER = TestTheVerdictCoversEveryWayToHideTheDomain.MASTER
    SCK = TestTheVerdictCoversEveryWayToHideTheDomain.SCK

    def _timed(self, **edits):
        return TestTheVerdictCoversEveryWayToHideTheDomain()._timed(**edits)

    @pytest.mark.parametrize("key, check", [("WHS(ns)", "hold"), ("WNS(ns)", "setup")])
    @pytest.mark.parametrize("index", [0, 1], ids=["core_clock_into_sck", "sck_into_core_clock"])
    def test_a_crossing_check_left_blank_fails(self, index, key, check):
        summary = self._timed()
        row = summary["inter"][index]
        row[key] = None
        assert _module().verdict(summary, expect_sck=1) == [
            f"{row['From Clock']} -> {row['To Clock']}: {key} has no number: the {check} check "
            f"across the SPI crossing is not timed"]

    def test_a_crossing_to_another_clock_is_held_to_the_same_rule(self):
        """Any crossing of an SCK clock, not only the one with the clock it is generated from."""
        other = {"From Clock": self.SCK, "To Clock": "hbm_aclk", "WNS(ns)": "4.0", "WHS(ns)": None}
        summary = self._timed(inter=self._timed()["inter"] + [other])
        assert _module().verdict(summary, expect_sck=1) == [
            f"{self.SCK} -> hbm_aclk: WHS(ns) has no number: the hold check across the SPI "
            f"crossing is not timed"]
        other["WHS(ns)"] = "-0.212"
        assert _module().verdict(summary, expect_sck=1) == [
            f"{self.SCK} -> hbm_aclk: WHS(ns) -0.212"]

    def test_the_fixture_is_the_fixed_builds_archived_summary(self):
        assert hashlib.sha256(FIXED_CU7.read_bytes()).hexdigest() == (
            "25956a300d7f628ab151653ec000c488631a54a83952f9f497522ba768bfa745")
        result = _gate(str(FIXED_CU7), "--expect-sck", "7")
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.rstrip().endswith("RESULT: PASS")

    @pytest.mark.parametrize("into_sck", [True, False],
                             ids=["core_clock_into_sck", "sck_into_core_clock"])
    def test_a_crossing_hold_check_blanked_in_the_fixed_builds_summary_fails(self, tmp_path,
                                                                              into_sck):
        sck = _sck_root(0)
        pair = (self.MASTER, sck) if into_sck else (sck, self.MASTER)
        text = FIXED_CU7.read_text()
        blanked = _blank_cells(text, "Inter Clock Table", pair, HOLD_CELLS)
        assert len(blanked) == len(text) and blanked != text
        path = tmp_path / "crossing_hold_false_pathed.summary.rpt"
        path.write_text(blanked)
        result = _gate(str(path), "--expect-sck", "7")
        assert result.returncode == 1, result.stdout + result.stderr
        fails = [line for line in result.stdout.splitlines() if line.startswith("FAIL:")]
        assert fails == [f"FAIL: {pair[0]} -> {pair[1]}: WHS(ns) has no number: the hold check "
                         f"across the SPI crossing is not timed"]
