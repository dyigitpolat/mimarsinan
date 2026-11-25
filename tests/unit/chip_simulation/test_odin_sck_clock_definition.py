"""[ODIN SPI clock domain] The SCK clock-definition check (U-SP4 L3): latency along the clock.

``scripts/hacc/check_sck_clock_definition.py`` reads the clock records a read-only probe of
a routed checkpoint writes (``usp4_probe.tcl``: the generated-clock sources the constraint
text resolves to, each SCK clock's properties, the fan-in of its SOURCE pin, the clock at
the SCK register's clock pin, the feedback paths with both clock paths, and the summary's
``check_timing``). Per SCK clock it requires:

  1. IS_GENERATED 1, IS_PROPAGATED 1, DIVIDE_BY 4, MASTER_CLOCK the kernel clock (the clock
     at ``sck_reg/C``), and SOURCE the one pin the text declares, on no clock buffer, whose
     timing fan-in reaches the master's source through no register;
  2. ``check_timing``: ``generated_clocks (0)``;
  3. the feedback's launch clock path, for setup and for hold, from the master's input port
     through SOURCE and its own cell's arc, a clock buffer and the core clock net to
     ``sck_reg/C``: no register's clock-to-output arc, every line a cell arc, a net or a pin,
     every net after the input buffer routed, and no latency or ideal-clock line.

It prints each SCK clock's defining file (FILE_NAME, LINE_NUMBER). With ``--own-build`` it
reads a build's own clocks: the probe in mode own (no ``read_xdc``; the clocks the checkpoint
restored, which carry no FILE_NAME), its SOURCE compared with what the build's packaged
``odin_spi_domain.xdc`` resolves to. A probe that re-reads a text replaces every SCK clock
(``[Constraints 18-619]``), so its records show how the text resolves in the routed netlist,
never what the build resolved at link; under ``--own-build`` such a record set fails by name.
Own-build read (MEASURED, the own-build probe): revision 3's own clock on S1-NC1, restored without
FILE_NAME or LINE_NUMBER, SOURCE ``sck_reg/C``, red by D-1 exactly as the control (the rev3 probe).

Red (MEASURED): revision 3's definitions re-applied to S1-NC1's routed checkpoint (the rev3
probe): SOURCE ``sck_reg/C``, whose fan-in holds ``core_en_q_reg`` and the ULP reset
register, and the feedback's setup launch path through the reset register (-1.573 ns, the
build's own figure). Green: the fix's text on the same checkpoint (the fix probe): SOURCE the
gate LUT's I0, feedback setup +5.077 ns at a source clock delay of 7.741 ns.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CHECK = REPO / "scripts" / "hacc" / "check_sck_clock_definition.py"
RECORDS = REPO / "tests" / "fixtures" / "odin_sck_clock_records"
REV3 = RECORDS / "s1nc1_rev3_probe"
FIX = RECORDS / "s1nc1_fix_probe"
OWN = RECORDS / "s1nc1_own_probe"

FIXTURE_SHA256 = {
    "s1nc1_fix_probe/feedback_paths.rpt": "b7dd8604db6160d4e190fbee9c5851f5b1df4964f6c48bfdf438e112964a4bb9",
    "s1nc1_fix_probe/master_clocks.txt": "ace9a96437b0c291c38e99543b6f59e088b46aab3cb8421f38d25ec1acf0e817",
    "s1nc1_fix_probe/resolved.txt": "73c848a7a0205274fa0da2cbd81b24fb1754b53e6409d87ebfa6f6bdee20b5e0",
    "s1nc1_fix_probe/sck_clock_properties.txt": "d08c69a8fc920743e6c128056ebedd222a5f72e62b084dc329a64e43a2dffe1a",
    "s1nc1_fix_probe/source_fanin.txt": "6ab753e275d236f6951c6af356f563f1796c8b0e4269a909d5c9b71ecd8ddee7",
    "s1nc1_fix_probe/timing_summary.head.rpt": "6b66d4e4a30359a6e5b54649ca6d078ac74e5abb557ffc7538f77df46698ac9e",
    "s1nc1_own_probe/feedback_paths.rpt": "1c34d2f33a602cd8c703e015adffdc5629d516abac41313e296baa852701320e",
    "s1nc1_own_probe/master_clocks.txt": "ace9a96437b0c291c38e99543b6f59e088b46aab3cb8421f38d25ec1acf0e817",
    "s1nc1_own_probe/probe_mode.txt": "d56bd3cd76bceb3b5f6e875beb27a6c45fcc673251870b2956cae7d0ee092d46",
    "s1nc1_own_probe/resolved.txt": "0c6f5bc6c01fdffdf94ed5a44f7b7049b4220f34703d4bfd7187ec09a347a140",
    "s1nc1_own_probe/sck_clock_properties.txt": "bad258ac7eceee7a18f2f581b95339d6e0c3872970015f1928b6ac523186af7a",
    "s1nc1_own_probe/source_fanin.txt": "2dde657459cf07ad3f2ec5db7221e3f8bdd6d0dc0b22a97181ed33b4d83ece24",
    "s1nc1_own_probe/timing_summary.head.rpt": "55f117e2161524110028204e727073586cb4beff881a0e8de5de2d00ead319c6",
    "s1nc1_own_probe/vivado.log.gz": "7f8f3cef01bdfefc174fa424befeef204cedda2a0a33296197130a8dd0e0ea00",
    "s1nc1_rev3_probe/feedback_paths.rpt": "5988bde0d9e87fb53d5788af77c0306324b92689f407056f43cb936ecde00b52",
    "s1nc1_rev3_probe/master_clocks.txt": "ace9a96437b0c291c38e99543b6f59e088b46aab3cb8421f38d25ec1acf0e817",
    "s1nc1_rev3_probe/resolved.txt": "0c6f5bc6c01fdffdf94ed5a44f7b7049b4220f34703d4bfd7187ec09a347a140",
    "s1nc1_rev3_probe/sck_clock_properties.txt": "b226e6e9b6bb1f09bb112609c004d4a4aa8ed29d1d40bc73a2bb6282beb0c4f5",
    "s1nc1_rev3_probe/source_fanin.txt": "2dde657459cf07ad3f2ec5db7221e3f8bdd6d0dc0b22a97181ed33b4d83ece24",
    "s1nc1_rev3_probe/timing_summary.head.rpt": "88f14bf9670479011cbd9c9cd7599ae6e0d13399f253c6472d0cb9259922f968",
}

KERNEL = "level0_i/ulp/odin_0/inst/kernel_i"
SCK = f"{KERNEL}/core_gen[0].spi_i/sck_reg/Q"
SCK_C = f"{KERNEL}/core_gen[0].spi_i/sck_reg/C"
GATE_I0 = f"{KERNEL}/core_gen[0].dut/scheduler_0/fifo_spike_0/FSM_onehot_phase[2]_i_3/I0"
RESET_REG = "level0_i/ulp/proc_sys_reset_freerun_slr0/U0/ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N"
ENABLE_REG = f"{KERNEL}/core_en_q_reg"


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-B", str(CHECK), *args], capture_output=True,
                          text=True)


def _copy(tmp_path: Path, src: Path = FIX) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def _edit(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text()
    assert old in text, (path.name, old)
    path.write_text(text.replace(old, new, count))


def _plain_log(folder: Path) -> Path:
    """The probe folder's log as a probe writes it: ``vivado.log`` in place of the fixture's
    ``vivado.log.gz`` (kept compressed in the tree, where ``*.log`` is ignored)."""
    stored = folder / "vivado.log.gz"
    plain = folder / "vivado.log"
    plain.write_bytes(gzip.decompress(stored.read_bytes()))
    stored.unlink()
    return plain


def _fails(result: "subprocess.CompletedProcess[str]") -> list[str]:
    return [ln[len("FAIL: "):] for ln in result.stdout.splitlines() if ln.startswith("FAIL: ")]


class TestTheFixturesAreTheProbeRecords:

    def test_each_record_is_the_probes_bytes(self):
        for name, digest in FIXTURE_SHA256.items():
            assert hashlib.sha256((RECORDS / name).read_bytes()).hexdigest() == digest, name


class TestRevision3sDefinitionIsRed:

    def test_the_rev3_source_and_launch_path_fail_by_name(self):
        result = _run(str(REV3), "--expect-sck", "1")
        assert result.returncode == 1, result.stdout + result.stderr
        fails = _fails(result)
        assert (f"{SCK}: the fan-in of SOURCE {SCK_C} holds register(s) {ENABLE_REG}, "
                f"{RESET_REG}") in fails
        assert (f"{SCK}: feedback setup: the launch clock path carries the clock-to-output "
                f"arc of {RESET_REG}/Q") in fails
        for check in ("setup", "hold"):
            assert (f"{SCK}: feedback {check}: the launch clock path does not leave SOURCE "
                    f"{SCK_C} by its own cell's arc") in fails
        assert result.stdout.rstrip().endswith("RESULT: FAIL")

    def test_the_rev3_properties_and_check_timing_are_not_what_fails(self):
        """Condition (2) reads 0 under revision 3 too: it guards an untraced form, not D-1."""
        result = _run(str(REV3))
        out = result.stdout
        assert "check_timing generated_clocks (0)" in out
        assert not [f for f in _fails(result) if "generated_clocks" in f or "IS_" in f
                    or "DIVIDE_BY" in f or "MASTER_CLOCK" in f]
        assert f"{SCK}: feedback setup: slack -1.573ns SCD 13.862ns" in out


class TestTheFixesDefinitionIsGreen:

    def test_the_fix_passes_with_its_latency_along_the_clock(self):
        result = _run(str(FIX), "--expect-sck", "1")
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert f"{SCK}: SOURCE {GATE_I0} (LUT3, group CLB, net type GLOBAL_CLOCK)" in out
        assert f"{SCK}: feedback setup: slack 5.077ns SCD 7.741ns" in out
        assert f"{SCK}: feedback hold: slack 2.526ns SCD 4.242ns" in out
        assert (f"{SCK}: feedback setup: through {GATE_I0} by LUT3 "
                "(Prop_H6LUT_SLICEL_I0_O), then BUFGCE, to " + SCK_C) in out
        assert out.rstrip().endswith("RESULT: PASS")


class TestEachConditionFailsOnItsOwn:

    @pytest.mark.parametrize("prop, value", [("IS_PROPAGATED", "0"), ("IS_GENERATED", "0"),
                                             ("DIVIDE_BY", "2")])
    def test_a_clock_property(self, tmp_path, prop, value):
        d = _copy(tmp_path)
        good = {"IS_PROPAGATED": "1", "IS_GENERATED": "1", "DIVIDE_BY": "4"}[prop]
        props = d / "sck_clock_properties.txt"
        lines = [ln if not ln.startswith(prop + " ") else ln[: ln.rindex(good)] + value
                 for ln in props.read_text().splitlines()]
        props.write_text("\n".join(lines) + "\n")
        result = _run(str(d))
        assert result.returncode == 1
        assert f"{SCK}: {prop} is {value}, not {good}" in _fails(result)

    def test_a_master_other_than_the_kernel_clock(self, tmp_path):
        d = _copy(tmp_path)
        _edit(d / "sck_clock_properties.txt", "MASTER_CLOCK       clock    true       io_clk_freerun_00_clk_p",
              "MASTER_CLOCK       clock    true       clk_kernel_00_unbuffered_net")
        result = _run(str(d))
        assert (f"{SCK}: MASTER_CLOCK is clk_kernel_00_unbuffered_net, not the kernel clock "
                "io_clk_freerun_00_clk_p") in _fails(result)

    def test_a_source_other_than_the_declared_pin(self, tmp_path):
        d = _copy(tmp_path)
        props = d / "sck_clock_properties.txt"
        text = props.read_text()
        props.write_text(text.replace(f"SOURCE             pin      true       {GATE_I0}",
                                      f"SOURCE             pin      true       {SCK_C}"))
        result = _run(str(d))
        assert f"{SCK}: SOURCE is {SCK_C}, not the declared {GATE_I0}" in _fails(result)

    def test_a_declared_source_on_a_clock_buffer(self, tmp_path):
        d = _copy(tmp_path)
        _edit(d / "resolved.txt", " ref LUT3 group CLB ref_pin I0", " ref BUFGCE group CLOCK ref_pin I0")
        result = _run(str(d))
        assert (f"{SCK}: the declared source {GATE_I0} is on a clock buffer (BUFGCE), not the "
                "gate") in _fails(result)

    def test_a_declared_source_that_resolves_to_no_pin_or_two(self, tmp_path):
        d = _copy(tmp_path)
        _edit(d / "resolved.txt", "source_count 1", "source_count 0")
        result = _run(str(d))
        assert f"{SCK}: the text's source resolves to 0 pins, not 1" in _fails(result)

    def test_a_register_in_the_source_fan_in(self, tmp_path):
        d = _copy(tmp_path)
        fanin = d / "source_fanin.txt"
        fanin.write_text(fanin.read_text() + f"cell {ENABLE_REG} ref FDRE group REGISTER seq 1\n")
        result = _run(str(d))
        assert f"{SCK}: the fan-in of SOURCE {GATE_I0} holds register(s) {ENABLE_REG}" in _fails(result)

    def test_a_fan_in_that_does_not_reach_the_masters_source(self, tmp_path):
        d = _copy(tmp_path)
        fanin = d / "source_fanin.txt"
        fanin.write_text("".join(ln + "\n" for ln in fanin.read_text().splitlines()
                                 if not ln.startswith("startpoint io_clk_freerun_00_clk_p")))
        result = _run(str(d))
        assert (f"{SCK}: the fan-in of SOURCE {GATE_I0} does not reach the master's source "
                "io_clk_freerun_00_clk_p") in _fails(result)

    def test_an_untraced_generated_clock(self, tmp_path):
        d = _copy(tmp_path)
        _edit(d / "timing_summary.head.rpt", "checking generated_clocks (0)",
              "checking generated_clocks (1)")
        result = _run(str(d))
        assert "check_timing reads generated_clocks (1), not (0)" in _fails(result)

    def test_a_latency_line_on_the_launch_clock_path(self, tmp_path):
        d = _copy(tmp_path)
        report = d / "feedback_paths.rpt"
        text = report.read_text()
        launch = text.index(f"(clock {SCK} ")  # the setup check's launch edge comes first
        at = text.index("\n", launch) + 1
        report.write_text(text[:at] + "                         clock source latency         2.000    22.000\n"
                          + text[at:])
        result = _run(str(d))
        assert (f"{SCK}: feedback setup: the launch clock path prints a latency or ideal "
                "clock line: clock source latency 2.000 22.000") in _fails(result)

    def test_an_unrouted_net_after_the_input_buffer(self, tmp_path):
        d = _copy(tmp_path)
        _edit(d / "feedback_paths.rpt", "net (fo=4726, routed)", "net (fo=4726)")
        result = _run(str(d))
        assert any(f.startswith(f"{SCK}: feedback setup: the launch clock path has an unrouted net")
                   for f in _fails(result)), result.stdout

    def test_a_missing_hold_path(self, tmp_path):
        d = _copy(tmp_path)
        report = d / "feedback_paths.rpt"
        text = report.read_text()
        report.write_text(text[: text.index("#### feedback hold")])
        result = _run(str(d))
        assert f"{SCK}: feedback hold: no path printed" in _fails(result)

    def test_a_missing_clock(self, tmp_path):
        d = _copy(tmp_path)
        props = d / "sck_clock_properties.txt"
        props.write_text(f"== (no clock on {SCK})\n")
        result = _run(str(d), "--expect-sck", "1")
        fails = _fails(result)
        assert f"no clock on {SCK}" in fails
        assert "0 SCK clock(s) in the records, expected 1" in fails

    def test_the_count_is_held_to_its_expectation(self, tmp_path):
        result = _run(str(FIX), "--expect-sck", "4")
        assert result.returncode == 1
        assert "1 SCK clock(s) in the records, expected 4" in _fails(result)

    def test_a_missing_record_is_not_read(self, tmp_path):
        d = _copy(tmp_path)
        (d / "source_fanin.txt").unlink()
        result = _run(str(d))
        assert result.returncode == 2
        assert "NOT A PROBE RECORD" in result.stdout


class TestEachClocksDefiningFileIsPrinted:

    def test_the_file_and_line_of_each_clock_are_printed(self):
        result = _run(str(FIX), "--expect-sck", "1")
        assert f"{SCK}: defined in /data/user/usp4/xdc_p1/fix2_nc1.xdc line 6" in result.stdout

    def test_a_restored_clock_prints_no_file(self):
        result = _run(str(OWN), "--expect-sck", "1")
        assert f"{SCK}: defined in (no FILE_NAME) line (no LINE_NUMBER)" in result.stdout


class TestTheOwnBuildReadRefusesAProbeText:
    """``--own-build`` reads a build's own clocks; the records of a probe that read a text are
    not that read, whatever their verdict otherwise."""

    @pytest.mark.parametrize("records, text", [
        (FIX, "/data/user/usp4/xdc_p1/fix2_nc1.xdc"),
        (REV3, "/data/user/usp4/xdc_explore/rev3_nc1.xdc"),
    ], ids=["the fix's text", "revision 3's text"])
    def test_a_probe_texts_records_fail_by_name(self, records, text):
        result = _run(str(records), "--expect-sck", "1", "--own-build")
        assert result.returncode == 1, result.stdout + result.stderr
        fails = _fails(result)
        assert ("the records are not a read of the build's own clocks: probe_mode.txt is "
                "absent, not 'mode own'") in fails
        assert ("the probe's vivado.log is absent: the restore of the build's constraints is "
                "not shown") in fails
        assert (f"{SCK}: FILE_NAME is {text}: the clock was defined by a text the probe read, "
                "not restored from the build's checkpoint") in fails

    def test_the_fix_texts_records_fail_only_by_those_names(self):
        result = _run(str(FIX), "--expect-sck", "1", "--own-build")
        fails = _fails(result)
        assert len(fails) == 3, fails
        assert not [f for f in fails
                    if "own clocks" not in f and "vivado.log" not in f and "FILE_NAME" not in f]


class TestTheBuildsOwnClocksAreRead:
    """The own-build read of S1-NC1's revision-3 checkpoint (the own-build probe, mode own, no
    ``read_xdc``): Vivado restored the build's SCK clock from the checkpoint (no FILE_NAME, no
    LINE_NUMBER), SOURCE ``sck_reg/C`` as the build's link resolved it, red by D-1 as built."""

    def test_revision_3s_own_clock_is_read_and_red_by_d1(self):
        result = _run(str(OWN), "--expect-sck", "1", "--own-build")
        assert result.returncode == 1, result.stdout + result.stderr
        out, fails = result.stdout, _fails(result)
        assert "read: the build's own clocks (--own-build); probe mode own" in out
        assert not [f for f in fails if "own clocks" in f or "vivado.log" in f
                    or "FILE_NAME" in f or "own read" in f]
        assert (f"{SCK}: the fan-in of SOURCE {SCK_C} holds register(s) {ENABLE_REG}, "
                f"{RESET_REG}") in fails
        assert (f"{SCK}: feedback setup: the launch clock path carries the clock-to-output "
                f"arc of {RESET_REG}/Q") in fails
        assert f"{SCK}: feedback setup: slack -1.573ns SCD 13.862ns" in out

    def test_a_plain_log_reads_as_the_stored_one(self, tmp_path):
        d = _copy(tmp_path, OWN)
        _plain_log(d)
        plain = _run(str(d), "--expect-sck", "1", "--own-build")
        stored = _run(str(OWN), "--expect-sck", "1", "--own-build")
        assert plain.stdout.splitlines()[1:] == stored.stdout.splitlines()[1:]

    def test_its_verdict_without_the_flag_is_the_controls(self):
        own = _fails(_run(str(OWN), "--expect-sck", "1"))
        rev3 = _fails(_run(str(REV3), "--expect-sck", "1"))
        assert own == rev3

    def test_a_clock_a_text_defined_fails_by_name(self, tmp_path):
        d = _copy(tmp_path, OWN)
        _edit(d / "sck_clock_properties.txt", "INPUT_JITTER ",
              "FILE_NAME          string   true       /data/user/usp4/xdc_p1/ctl_nc1.xdc\nINPUT_JITTER ")
        result = _run(str(d), "--expect-sck", "1", "--own-build")
        assert (f"{SCK}: FILE_NAME is /data/user/usp4/xdc_p1/ctl_nc1.xdc: the clock was defined "
                "by a text the probe read, not restored from the build's checkpoint") in _fails(result)

    def test_a_record_of_mode_apply_fails_by_name(self, tmp_path):
        d = _copy(tmp_path, OWN)
        (d / "probe_mode.txt").write_text("mode apply\n")
        result = _run(str(d), "--expect-sck", "1", "--own-build")
        assert ("the records are not a read of the build's own clocks: probe_mode.txt reads "
                "'mode apply', not 'mode own'") in _fails(result)

    def test_an_own_read_of_another_text_fails_by_name(self, tmp_path):
        d = _copy(tmp_path, OWN)
        _edit(d / "probe_mode.txt", "hw_nc1/odin_spi_domain.xdc", "hw_nc1/ctl_nc1.xdc")
        result = _run(str(d), "--expect-sck", "1", "--own-build")
        assert ("the own read's text is /data/user/survey_r0/builds/S1-NC1/odin_power_overlay/"
                "build/hacc/hw_nc1/ctl_nc1.xdc, not the build's packaged odin_spi_domain.xdc"
                in _fails(result))

    def test_a_clock_replaced_after_the_restore_fails_by_name(self, tmp_path):
        d = _copy(tmp_path, OWN)
        log = _plain_log(d)
        log.write_text(log.read_text() + f"WARNING: [Constraints 18-619] A clock with name '{SCK}' "
                       "already exists, overwriting the previous clock with the same name.\n")
        result = _run(str(d), "--expect-sck", "1", "--own-build")
        assert (f"the probe's vivado.log shows clock {SCK} replaced after the restore "
                "([Constraints 18-619])") in _fails(result)

    def test_a_log_without_the_restore_fails_by_name(self, tmp_path):
        d = _copy(tmp_path, OWN)
        _edit(_plain_log(d), "Restoring constraints from binary archive", "Restoring nothing")
        result = _run(str(d), "--expect-sck", "1", "--own-build")
        assert ("the probe's vivado.log shows no 'Restoring constraints from binary archive'"
                in _fails(result))


class TestTheStoredSummaryIsRead:
    """The probe records keep the full summary as ``timing_summary.rpt.gz``."""

    def test_a_gzip_summary_reads_as_the_plain_one(self, tmp_path):
        d = _copy(tmp_path)
        head = d / "timing_summary.head.rpt"
        (d / "timing_summary.rpt.gz").write_bytes(gzip.compress(head.read_bytes(), mtime=0))
        head.unlink()
        stored = _run(str(d), "--expect-sck", "1")
        plain = _run(str(FIX), "--expect-sck", "1")
        assert stored.returncode == 0, stored.stdout + stored.stderr
        assert stored.stdout.splitlines()[1:] == plain.stdout.splitlines()[1:]
