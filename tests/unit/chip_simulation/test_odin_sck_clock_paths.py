"""[ODIN SPI clock domain] The SCK clock-path check (U-SP4 L1): no side input in an SCK latency.

``scripts/hacc/check_sck_clock_paths.py`` reads the Timing Details of a routed summary.
For every printed path whose launch or capture clock is an SCK clock, in every table,
it reads both clock paths and fails when either holds a register's clock-to-output arc
other than the SCK register's own (the generated clock's source pin). It counts the
paths that print no clock path, so an empty read is seen, and fails a read with no SCK
path at all.

Red, MEASURED (survey build lock revision 2, ``clock_path_read``): S1-NC1's SCK
sections, 41 clock paths with the ULP reset register in five sets of checks; the
SCK-to-core sections of S1-NC4 and of the seven-unit build, the feedback's setup launch clock
path in 4 of 4 and 7 of 7 domains. The fixtures are byte copies of the diagnosis records
(``tests/fixtures/odin_routed_timing/PROVENANCE.md``). Green: SCK paths whose clock paths
follow the clock.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CHECK = REPO / "scripts" / "hacc" / "check_sck_clock_paths.py"
FIXTURES = REPO / "tests" / "fixtures" / "odin_routed_timing"
S1NC1 = FIXTURES / "s1nc1_sck_sections.rpt.gz"
S1NC4 = FIXTURES / "s1nc4_sck_to_core_sections.rpt.gz"
FIXED_CU7 = FIXTURES / "fixed_cu7_sck_to_core_sections.rpt.gz"

#: sha256 of each fixture's bytes and of its uncompressed text (the diagnosis records).
FIXTURE_SHA256 = {
    S1NC1.name: ("b3bad7265ab4b7ffc8612417db869a56a72cffc1e549c5e67c50b7acac015c27",
                 "c8523936b03b7d77bedb118116294f8716edf46139462ef9a34023add79159fe"),
    S1NC4.name: ("2d19c6f32038f1a9d9a5f0b6986a4c7915f54e2f666641cbb2e2b1c712bd1879",
                 "811269ddb4219c918d429a0eca9a0ce0d816c9707813120bc847185452d2d0cc"),
    FIXED_CU7.name: ("48369320d15660171bbd74ffb9feddfd6bb0c50f9d981a673fdb72f78fd1a7a2",
                     "fc2f559493b0f8c235d8ca99f82f57cf649223714cd4beeb10756572d73e07c3"),
}

RESET_REG = ("level0_i/ulp/proc_sys_reset_freerun_slr0/U0/"
             "ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N/Q")


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-B", str(CHECK), *args], capture_output=True,
                          text=True)


def _module():
    spec = importlib.util.spec_from_file_location("check_sck_clock_paths_under_test", CHECK)
    assert spec is not None and spec.loader is not None, f"{CHECK} is missing"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _paths(path: Path):
    mod = _module()
    return mod.parse_paths(mod.read_text(path), path.name)


def _sck(unit: int, core: int = 0) -> str:
    return f"level0_i/ulp/odin_{unit}/inst/kernel_i/core_gen[{core}].spi_i/sck_reg/Q"


# A synthetic summary in the routed summary's own line shapes: one SCK domain whose clock
# paths follow the clock (the platform clock, its input and global buffers, the gate's
# clock input I0, the core-clock buffer, the core clock net), with one feedback setup path
# (SCK to core) and one MOSI hold path into the slave (core to SCK), where the SCK register's
# own clock-to-output arc sits on the capture clock path, as a generated clock's arc does.
SCK = "level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/sck_reg/Q"
GATE = "level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/scheduler_0/fifo_spike_0/FSM_onehot_phase[2]_i_3"
BUFC = "level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/scheduler_0/fifo_spike_0/ap_clk_0_BUFGCE_inst"
SCKC = "level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/sck_reg/C"

CLOCK_ARC_PATH = f"""\
    BK43                                              0.000    20.000 r  static                              io_clk_freerun_00_clk_p (IN)
                         net (fo=0)                   0.100    20.100    static                              level0_i/blp/blp_i/freerun_clk/freerun_clk_ibufds/U0/USE_IBUFDS.GEN_IBUFDS[0].IBUFDS_I/I
    HPIOBDIFFINBUF_X0Y34 DIFFINBUF (Prop_DIFFINBUF_HPIOBDIFFINBUF_DIFF_IN_P_O)
                                                      0.777    20.877 r  static         pblock_blp           level0_i/blp/blp_i/freerun_clk/freerun_clk_ibufds/U0/USE_IBUFDS.GEN_IBUFDS[0].IBUFDS_I/DIFFINBUF_INST/O
                         net (fo=1, routed)           0.347    21.224    static                              level0_i/blp/blp_i/freerun_clk/bufg_div/U0/BUFGCE_I[0]
    BUFGCE_DIV_X0Y4                                                   r  static         pblock_blp           level0_i/blp/blp_i/freerun_clk/bufg_div/U0/USE_BUFGCE_DIV2.GEN_BUFGCE_DIV2[0].BUFGCE_DIV2_I/I
    BUFGCE_DIV_X0Y4      BUFGCE_DIV (Prop_BUFGCE_DIV_I_O)
                                                      0.151    21.375 r  static         pblock_blp           level0_i/blp/blp_i/freerun_clk/bufg_div/U0/USE_BUFGCE_DIV2.GEN_BUFGCE_DIV2[0].BUFGCE_DIV2_I/O
                         net (fo=4726, routed)        3.101    24.476    boundary                            level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/scheduler_0/fifo_spike_0/ap_clk
    SLICE_X117Y31                                                     r  reconfigurable pblock_dynamic_region
                                                                                                             {GATE}/I0
    SLICE_X117Y31        LUT3 (Prop_H6LUT_SLICEL_I0_O)
                                                      0.095    24.571 r  reconfigurable pblock_dynamic_region
                                                                                                             {GATE}/O
                         net (fo=1, routed)           0.291    24.862    reconfigurable                      level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/scheduler_0/fifo_spike_0/ap_clk_0_BUFG
    BUFGCE_X0Y1                                                       r  reconfigurable pblock_dynamic_region
                                                                                                             {BUFC}/I
    BUFGCE_X0Y1          BUFGCE (Prop_BUFCE_BUFGCE_I_O)
                                                      0.024    24.886 r  reconfigurable pblock_dynamic_region
                                                                                                             {BUFC}/O
    X0Y0 (CLOCK_ROOT)    net (fo=3986, routed)        1.958    26.844    reconfigurable                      level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/core_clk
    SLICE_X20Y22                                                      r  reconfigurable pblock_dynamic_region
                                                                                                             {SCKC}
"""

SIDE_ARC = """\
    SLICE_X114Y142       FDRE (Prop_EFF_SLICEM_C_Q)
                                                      0.076    24.641 f  reconfigurable pblock_dynamic_region/pblock_dynamic_SLR0
                                                                                                             level0_i/ulp/proc_sys_reset_freerun_slr0/U0/ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N/Q
"""

GEN_ARC = f"""\
    SLICE_X20Y22         FDRE (Prop_HFF_SLICEL_C_Q)
                                                      0.079    26.923 f  reconfigurable pblock_dynamic_region
                                                                                                             {SCK}
                         net (fo=1, routed)           2.177    29.100    reconfigurable                      level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/core_gen[0].sck_w
"""


def _summary(launch_extra: str = "", capture_extra: str = "") -> str:
    return f"""\
------------------------------------------------------------------------------------------------
| Timing Details
| --------------
------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------
From Clock:  {SCK}
  To Clock:  io_clk_freerun_00_clk_p

Setup :            0  Failing Endpoints,  Worst Slack        4.600ns,  Total Violation        0.000ns
---------------------------------------------------------------------------------------------------


Max Delay Paths
--------------------------------------------------------------------------------------
Slack (MET) :             4.600ns  (required time - arrival time)
  Source:                 {SCK}
  Destination:            level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/sck_reg/D
  Path Group:             io_clk_freerun_00_clk_p
  Path Type:              Setup (Max at Slow Process Corner)

    Location             Delay type                Incr(ns)  Path(ns)    Partition      PBlock               Netlist Resource(s)
  -------------------------------------------------------------------    -------------------------------------------------------
                         (clock {SCK} fall edge)
                                                     20.000    20.000 f
{launch_extra}{CLOCK_ARC_PATH}  -------------------------------------------------------------------    -------------------------------------------------------
    SLICE_X20Y22         FDRE (Prop_HFF_SLICEL_C_Q)
                                                      0.079    26.923 f  reconfigurable pblock_dynamic_region
                                                                                                             {SCK}
  -------------------------------------------------------------------    -------------------------------------------------------

                         (clock io_clk_freerun_00_clk_p rise edge)
                                                     30.000    30.000 r
{CLOCK_ARC_PATH}                         clock pessimism              0.432    37.124
                         clock uncertainty           -0.035    37.089

Slack (MET) :             4.900ns  (required time - arrival time)
  Source:                 level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/spi_slave_0/CTRL_SPI_ADDR_reg[11]/C
  Destination:            level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/scheduler_0/fifo_burst/mem_reg/D
  Path Group:             io_clk_freerun_00_clk_p
  Path Type:              Setup (Max at Slow Process Corner)
  Timing Exception:       MaxDelay Path 20.000ns -datapath_only

    Location             Delay type                Incr(ns)  Path(ns)    Partition      PBlock               Netlist Resource(s)
  -------------------------------------------------------------------    -------------------------------------------------------
    SLICE_X21Y22         FDCE                         0.000     0.000 r  reconfigurable pblock_dynamic_region
                                                                                                             level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/spi_slave_0/CTRL_SPI_ADDR_reg[11]/C
  -------------------------------------------------------------------    -------------------------------------------------------


---------------------------------------------------------------------------------------------------
From Clock:  io_clk_freerun_00_clk_p
  To Clock:  {SCK}

Hold  :            0  Failing Endpoints,  Worst Slack       19.000ns,  Total Violation        0.000ns
---------------------------------------------------------------------------------------------------


Min Delay Paths
--------------------------------------------------------------------------------------
Slack (MET) :             19.000ns  (arrival time - required time)
  Source:                 level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/mosi_reg/C
  Destination:            level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/spi_slave_0/spi_shift_reg_in_reg[0]/D
  Path Group:             {SCK}
  Path Type:              Hold (Min at Fast Process Corner)
  Timing Exception:       MultiCycle Path   Hold  -start 3

    Location             Delay type                Incr(ns)  Path(ns)    Partition      PBlock               Netlist Resource(s)
  -------------------------------------------------------------------    -------------------------------------------------------
                         (clock io_clk_freerun_00_clk_p rise edge)
                                                      0.000     0.000 r
{CLOCK_ARC_PATH}  -------------------------------------------------------------------    -------------------------------------------------------
    SLICE_X20Y23         FDRE (Prop_AFF_SLICEL_C_Q)
                                                      0.080     6.924 r  reconfigurable pblock_dynamic_region
                                                                                                             level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].spi_i/mosi_reg/Q
  -------------------------------------------------------------------    -------------------------------------------------------

                         (clock {SCK} rise edge)
                                                     -30.000  -30.000 r
{capture_extra}{CLOCK_ARC_PATH}{GEN_ARC}    SLICE_X21Y22                                                      r  reconfigurable pblock_dynamic_region
                                                                                                             level0_i/ulp/odin_0/inst/kernel_i/core_gen[0].dut/spi_slave_0/spi_shift_reg_in_reg[0]/C
                         clock pessimism             -0.351   -22.000
"""


class TestTheFixturesAreTheDiagnosisRecords:

    def test_each_fixture_is_its_record(self):
        for path in (S1NC1, S1NC4, FIXED_CU7):
            raw = path.read_bytes()
            packed, plain = FIXTURE_SHA256[path.name]
            assert hashlib.sha256(raw).hexdigest() == packed, path.name
            assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == plain, path.name


class TestTheRevision3BuildsAreRed:

    def test_s1nc1_carries_the_reset_register_on_41_clock_paths_in_five_sets(self):
        paths = _paths(S1NC1)
        assert len(paths) == 71
        assert sum(1 for p in paths if not p.clock_paths) == 9
        assert sum(1 for p in paths if len(p.clock_paths) == 2) == 62
        sets = {}
        for p in paths:
            for which in ("launch", "capture"):
                regs = p.side_registers(which)
                if regs:
                    assert regs == [RESET_REG], (p.start, p.end, which, regs)
                    key = (p.from_clock == _sck(0), p.to_clock == _sck(0), p.group,
                           p.path_type, which)
                    sets[key] = sets.get(key, 0) + 1
        assert sum(sets.values()) == 41
        assert sets == {
            (True, True, "(clock pair)", "Setup", "launch"): 10,
            (True, True, "(clock pair)", "Hold", "capture"): 10,
            (True, False, "(clock pair)", "Setup", "launch"): 1,
            (False, True, "(clock pair)", "Hold", "capture"): 10,
            (False, True, "**async_default**", "Removal", "capture"): 10,
        }
        # The one SCK-to-core setup path with a side register is the feedback.
        feedback = [p for p in paths if p.from_clock == _sck(0) and p.side_registers("launch")
                    and p.to_clock != _sck(0)]
        assert [(p.start, p.end, p.slack) for p in feedback] == [
            (_sck(0), _sck(0)[:-1] + "D", "-1.573ns")]

    def test_s1nc1_fails_by_name(self):
        result = _run(str(S1NC1))
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout
        assert "paths printed: 71; with no clock path: 9; with both clock paths: 62" in out
        assert "clock paths with a side register: launch 11, capture 30, total 41" in out
        assert f"FAIL: 11 launch clock path(s) of SCK paths carry the clock-to-output arc of {RESET_REG}" in out
        assert f"FAIL: 30 capture clock path(s) of SCK paths carry the clock-to-output arc of {RESET_REG}" in out
        assert out.rstrip().endswith("RESULT: FAIL")

    @pytest.mark.parametrize("fixture, units, cores, slaves", [
        (S1NC4, 1, 4, 36), (FIXED_CU7, 7, 1, 63)])
    def test_every_closed_domain_carries_the_arc_on_its_feedback_setup(self, fixture, units,
                                                                       cores, slaves):
        paths = _paths(fixture)
        domains = {_sck(u, c) for u in range(units) for c in range(cores)}
        setup = [p for p in paths if p.start == p.from_clock and p.path_type == "Setup"]
        hold = [p for p in paths if p.start == p.from_clock and p.path_type == "Hold"]
        assert {p.from_clock for p in setup} == domains
        assert all(p.end == p.from_clock[:-1] + "D" for p in setup + hold)
        assert all(p.side_registers("launch") for p in setup)
        assert not any(p.side_registers("capture") for p in setup)
        assert {p.from_clock for p in hold} == domains
        assert not any(p.side_registers("launch") or p.side_registers("capture") for p in hold)
        assert sum(1 for p in paths if not p.clock_paths) == slaves
        side = {r for p in setup for r in p.side_registers("launch")}
        assert side and all(r.endswith("core_en_q_reg/Q") or "proc_sys_reset" in r for r in side)
        result = _run(str(fixture), "--expect-sck", str(units * cores))
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"SCK clocks: {units * cores}" in result.stdout


class TestClockPathsAlongTheClockAreGreen:

    def test_a_domain_timed_along_the_clock_passes(self, tmp_path):
        report = tmp_path / "clean.rpt"
        report.write_text(_summary())
        result = _run(str(report), "--expect-sck", "1")
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert "paths printed: 3; with no clock path: 1; with both clock paths: 2" in out
        assert "clock paths with a side register: launch 0, capture 0, total 0" in out
        assert out.rstrip().endswith("RESULT: PASS")

    def test_the_sck_registers_own_arc_is_not_a_side_register(self, tmp_path):
        report = tmp_path / "clean.rpt"
        report.write_text(_summary())
        mod = _module()
        hold = [p for p in mod.parse_paths(report.read_text()) if p.path_type == "Hold"]
        assert len(hold) == 1
        assert hold[0].capture is not None
        assert hold[0].capture.sequential_outputs() == [SCK]
        assert hold[0].side_registers("capture") == []

    def test_gzip_and_section_copies_read_alike(self, tmp_path):
        plain = tmp_path / "clean.rpt"
        plain.write_text(_summary())
        packed = tmp_path / "clean.rpt.gz"
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        copy = tmp_path / "copy.rpt"
        copy.write_text("".join("#### " + ln + "\n" for ln in _summary().splitlines()))
        outs = [_run(str(p)).stdout.split("\n", 0)[0] for p in (plain, packed, copy)]
        lines = [[ln.split(" | ", 1)[1] for ln in o.splitlines() if " | sec " in ln] for o in outs]
        assert lines[0] == lines[1] == lines[2] and len(lines[0]) == 3


class TestASideRegisterAnywhereIsRed:

    def test_on_the_launch_clock_path(self, tmp_path):
        report = tmp_path / "launch.rpt"
        report.write_text(_summary(launch_extra=SIDE_ARC))
        result = _run(str(report))
        assert result.returncode == 1
        assert f"FAIL: 1 launch clock path(s) of SCK paths carry the clock-to-output arc of {RESET_REG}" in result.stdout

    def test_on_the_capture_clock_path(self, tmp_path):
        report = tmp_path / "capture.rpt"
        report.write_text(_summary(capture_extra=SIDE_ARC))
        result = _run(str(report))
        assert result.returncode == 1
        assert f"FAIL: 1 capture clock path(s) of SCK paths carry the clock-to-output arc of {RESET_REG}" in result.stdout

    def test_another_sck_registers_arc_is_a_side_register(self, tmp_path):
        """Only the path's own SCK clocks' source pins are their own arcs."""
        other = GEN_ARC.replace("odin_0", "odin_1").split("\n                         net")[0] + "\n"
        report = tmp_path / "other.rpt"
        report.write_text(_summary(launch_extra=other))
        result = _run(str(report))
        assert result.returncode == 1
        assert "clock-to-output arc of " + SCK.replace("odin_0", "odin_1") in result.stdout


class TestAnEmptyReadIsNeverAPass:

    def test_no_sck_section_fails(self, tmp_path):
        text = _summary().replace(SCK, "level0_i/ulp/odin_0/inst/kernel_i/other_reg/Q")
        report = tmp_path / "none.rpt"
        report.write_text(text)
        result = _run(str(report))
        assert result.returncode == 1
        assert "FAIL: no path of an SCK clock is printed" in result.stdout
        assert "paths printed: 0" in result.stdout

    def test_the_sck_count_is_held_to_its_expectation(self, tmp_path):
        report = tmp_path / "clean.rpt"
        report.write_text(_summary())
        result = _run(str(report), "--expect-sck", "4")
        assert result.returncode == 1
        assert "FAIL: 1 SCK clock(s) in the printed paths, expected 4" in result.stdout

    def test_a_file_without_timing_details_is_not_read(self, tmp_path):
        report = tmp_path / "head.rpt"
        report.write_text("| Design Timing Summary\n| ---------------------\nWNS(ns) 0.1\n")
        result = _run(str(report))
        assert result.returncode == 2
        assert "NOT A TIMING SUMMARY" in result.stdout
