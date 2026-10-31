"""[ODIN SPI clock domain] Every stock build times the SPI slave's clock, by data, per core.

Each vendored ODIN core is programmed through its SPI slave, and the slave is
clocked by SCK: the output of a register (``sck``) in the kernel's SPI master
(``hw/fpga/kernel/odin_spi_master.v``), toggling at CLK/4 on the gated core
clock. Until this change no build declared a clock on it, so every routed
summary listed 381 unclocked register pins and 790 unconstrained endpoints per
core, and on a seven-unit build the fabric-routed SCK domains carried same-edge
hold races nobody had timed (``spi_triage.md``). The fix is a constraint set,
the RTL byte-identical. The claims gated here, off-cluster (stub Vivado, v++):

  * ``build_xclbn.sh`` writes the constraint set for every core of the kernel it
    packages (NC is data), for a fabric whose cores carry the vendored SPI slave,
    and for no other: one generated clock per SCK register (divide by 4 of the
    register's own clock, defined at the core-clock gate's clock input so its
    latency follows the clock arc and no side input of the gate: U-SP4, survey
    build lock revision 2, defect D-1; the rules of that text, and the forms it
    may never carry, are checked by name per core at NC 1, 2, 4 and 7), a hold
    multicycle of 3 core cycles into the domain
    (MOSI is set one core cycle before SCK rises and held three after it), a
    data-path bound of two core periods out of the slave into the core (its
    events pass two-flop synchronizers, its buses are stable when the core acts
    on the synchronized event), ASYNC_REG on those synchronizers, and a global
    buffer asked for on every kernel clock (each SCK net, and once per kernel the
    gated core clock, so no clock is left to opt_design's eight-buffer budget);
  * the set is scoped to the kernel top, so it names no compute unit: it applies
    to each unit the link instantiates, at any unit count, byte for byte the same;
  * the packaging adds it for implementation only, processed late (after the
    shell's clocks exist), and the build refuses before the link unless the
    packaged core carries it so;
  * every name the set uses is the RTL's own, so a rename in the RTL fails here
    and not hours into a link.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HACC = REPO / "scripts" / "hacc"
STUBS = HACC / "selftest" / "stubs" / "vitis_bin"
XDC_NAME = "odin_spi_domain.xdc"

BUILD_ENV = ("ODIN_COMPUTE_UNITS", "ODIN_CHIP", "ODIN_CU_SLRS", "ODIN_CARD",
             "ODIN_KERNEL_CORES", "ODIN_LOG", "ODIN_STUB_PACKAGED_NC",
             "ODIN_STUB_PACKAGED_XDC", "ODIN_VXX_CONFIG")

#: The synchronizers through which the core reads the slave's single-bit outputs
#: (vendored controller.v and scheduler.v), as synthesis names their registers.
SYNC_REGS = (
    "controller_0/SPI_GATE_ACTIVITY_sync_int_reg", "controller_0/SPI_GATE_ACTIVITY_sync_reg",
    "controller_0/CTRL_READBACK_EVENT_sync_int_reg", "controller_0/CTRL_READBACK_EVENT_sync_reg",
    "controller_0/CTRL_PROG_EVENT_sync_int_reg", "controller_0/CTRL_PROG_EVENT_sync_reg",
    "scheduler_0/SPI_OPEN_LOOP_sync_int_reg", "scheduler_0/SPI_OPEN_LOOP_sync_reg",
)


def _stage_build_tree(root: Path) -> Path:
    """The files build_xclbn.sh reads, laid out as a package lays them out."""
    hacc = root / "scripts" / "hacc"
    hacc.mkdir(parents=True)
    for name in ("build_xclbn.sh", "toolchain.sh", "cards.sh", "chips.sh",
                 "odin_u55c.cfg", "odin_u250.cfg", "gen_link_cfg.py"):
        shutil.copyfile(HACC / name, hacc / name)
    shutil.copyfile(HACC / "package" / "sbatch" / "odin_build.sbatch",
                    hacc / "odin_build.sbatch")
    (hacc / "gen_kernel_xml.py").write_text(
        "import sys\nout = sys.argv[sys.argv.index('--output') + 1]\n"
        "open(out, 'w').write('<root/>\\n')\nprint('[stub] kernel.xml')\n")
    for relative in ("hw/fpga/kernel", "hw/fpga/mem", "hw/vendor/odin/src", "hw/gen/chips"):
        shutil.copytree(REPO / relative, root / relative)
    for script in hacc.glob("*.sh"):
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return root


def _stub_toolchain(root: Path) -> dict:
    vitis = root / "tools" / "Vitis" / "2022.2"
    (vitis / "bin").mkdir(parents=True)
    for tool in ("vivado", "v++"):
        shutil.copyfile(STUBS / tool, vitis / "bin" / tool)
        (vitis / "bin" / tool).chmod(0o755)
    (vitis / "settings64.sh").write_text(
        f'export XILINX_VITIS="{vitis}"\nexport PATH="{vitis}/bin:${{PATH}}"\n')
    xrt = root / "xrt"
    xrt.mkdir()
    (xrt / "setup.sh").write_text("")
    return {"XILINX_ROOT": str(root / "tools"), "XRT_ROOT": str(xrt),
            "ODIN_PLATFORM_ROOT": str(root / "no_platforms_here"),
            "ODIN_STUB_VXX_TRACE": str(root / "vxx.trace")}


def _build(tmp_path: Path, **extra: str) -> tuple:
    root = _stage_build_tree(tmp_path / "pkg")
    run_env = {k: v for k, v in os.environ.items() if k not in BUILD_ENV}
    run_env.update(_stub_toolchain(tmp_path))
    run_env.update(extra)
    result = subprocess.run(["bash", "scripts/hacc/build_xclbn.sh", "hw"], capture_output=True,
                            text=True, cwd=str(root), env=run_env)
    return root, result


def _leaf(cores: str = "1", units: str = "1", chip_suffix: str = "") -> str:
    leaf = f"hw_nc{cores}{chip_suffix}"
    return leaf if units == "1" else f"{leaf}_cu{units}"


def _code(text: str) -> list:
    return [line for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _sck(c: int, pin: str) -> str:
    return f"[get_pins {{kernel_i/core_gen[{c}].spi_i/sck_reg/{pin}}}]"


def _gate_clock_input(c: int) -> str:
    """The core-clock gate's clock input, found from core c's SCK register (U-SP4).

    The gate is the LUT in the timing fan-in of ``sck_reg/C`` (the cone is the same before
    and after opt_design puts the requested global buffer on the gated net, and holds no
    other LUT); its clock input is its input whose net segment is a clock net (a global
    clock in the linked design, a local clock where the kernel clock is a port, as in the
    kernel's own synthesized netlist; the enable and reset nets are signals in both); the
    pin is the one leaf, non-clock input on that segment. Only the clock arc reaches it, so
    a generated clock defined there takes its latency along the clock (probe 71248 on
    S1-NC1: the LUT3's I0, never the inserted BUFGCE; the link-stage read of S1-NC1's
    kernel netlist: the same LUT, its I0 on a LOCAL_CLOCK net)."""
    gate = ("[get_cells -filter {PRIMITIVE_SUBGROUP == LUT} "
            f"[all_fanin -flat -only_cells {_sck(c, 'C')}]]")
    return ("[get_pins -filter {IS_LEAF && DIRECTION == IN && !IS_CLOCK} -of_objects "
            "[get_nets -filter {TYPE =~ *_CLOCK} -of_objects "
            f"[get_pins -filter {{DIRECTION == IN}} -of_objects {gate}]]]")


def _expected_core_lines(c: int) -> list:
    master = f"[get_clocks -of_objects {_sck(c, 'C')}]"
    sck = f"[get_clocks -of_objects {_sck(c, 'Q')}]"
    slave = f"[get_cells -filter {{IS_SEQUENTIAL}} {{kernel_i/core_gen[{c}].dut/spi_slave_0/*}}]"
    syncs = " ".join(f"kernel_i/core_gen[{c}].dut/{reg}" for reg in SYNC_REGS)
    return [
        f"create_generated_clock -source {_gate_clock_input(c)} -divide_by 4 {_sck(c, 'Q')}",
        f"set_multicycle_path -hold -start 3 -from {master} -to {sck}",
        f"set_max_delay -datapath_only -from {slave} -to {master} "
        f"[expr {{2.0 * [get_property PERIOD {master}]}}]",
        f"set_property CLOCK_BUFFER_TYPE BUFG [get_nets -of_objects {_sck(c, 'Q')}]",
        f"set_property ASYNC_REG TRUE [get_cells {{{syncs}}}]",
    ]



#: The kernel's gated core clock, asked for a global buffer once per kernel, on the net
#: at its LEAF DRIVER (the gating LUT): the SPI master of core 0 (present at every NC)
#: is clocked by it, like every core. Asked on the hierarchical segment at the master's
#: clock pin, the buffer covers only the master's 58 clock pins and the LUT keeps
#: driving the other 3,929 (probe 71068; build 71057, Place 30-568).
CORE_CLOCK_LINE = (
    "set_property CLOCK_BUFFER_TYPE BUFG [get_nets -of_objects [get_pins -leaf "
    f"-filter {{DIRECTION == OUT}} -of_objects [get_nets -of_objects {_sck(0, 'C')}]]]")

# ---------------------------------------------------------------------------
# The constraint set: one block per core, from NC alone
# ---------------------------------------------------------------------------


class TestTheBuildWritesTheSpiConstraintPerCore:

    @pytest.mark.parametrize("cores", ["1", "2", "4", "7"])
    def test_every_core_gets_its_clock_and_its_exceptions(self, tmp_path, cores):
        root, result = _build(tmp_path, ODIN_KERNEL_CORES=cores)
        assert result.returncode == 0, result.stdout + result.stderr
        xdc = root / "build" / "hacc" / _leaf(cores) / XDC_NAME
        assert xdc.is_file(), result.stdout
        expected = [CORE_CLOCK_LINE] + [
            line for c in range(int(cores)) for line in _expected_core_lines(c)]
        assert _code(xdc.read_text()) == expected
        assert f"[hacc-build] spi xdc  : {cores} SPI clock domain(s)" in result.stdout

    def test_the_set_names_no_unit_and_is_the_same_at_every_unit_count(self, tmp_path):
        texts = []
        for units in ("1", "7"):
            root, result = _build(tmp_path / f"cu{units}", ODIN_COMPUTE_UNITS=units)
            assert result.returncode == 0, result.stdout + result.stderr
            texts.append((root / "build" / "hacc" / _leaf("1", units) / XDC_NAME).read_bytes())
        assert texts[0] == texts[1]
        text = texts[0].decode()
        # Scoped to the kernel top: every object is a path below it, none above it.
        assert not re.search(r"odin_[0-9]|level0|ulp/|-hierarchical", text)
        for line in _code(text):
            groups = [g for g in re.findall(r"\{([^{}]*)\}", line) if "/" in g]
            assert groups, line
            for group in groups:
                assert all(part.startswith("kernel_i/") for part in group.split()), line

    def test_the_core_clock_is_asked_for_once_whatever_the_core_count(self, tmp_path):
        """opt_design's BUFG optimization inserts at most eight buffers per build, and a
        buffer asked for by property counts toward them: at N = 7, SCK requests alone
        left six units' gated core clocks unbuffered (build 71055). Every kernel clock
        is therefore asked for by property, the core clock once per kernel."""
        for cores in ("1", "4"):
            root, result = _build(tmp_path / f"nc{cores}", ODIN_KERNEL_CORES=cores)
            assert result.returncode == 0, result.stdout + result.stderr
            lines = _code((root / "build" / "hacc" / _leaf(cores) / XDC_NAME).read_text())
            requests = [line for line in lines if "CLOCK_BUFFER_TYPE BUFG" in line]
            assert requests.count(CORE_CLOCK_LINE) == 1
            assert len(requests) == 1 + int(cores)

    def test_the_hold_relaxation_and_the_divider_are_the_masters_own(self):
        """SCK = CLK/4; MOSI changes one CLK before SCK rises and three CLK after it."""
        master = (REPO / "hw" / "fpga" / "kernel" / "odin_spi_master.v").read_text()
        assert "SCK = CLK/4" in master
        assert "output reg         sck" in master
        phases = re.search(r"2'd0: begin\s*mosi\s*<= frame\[39\];\s*sck\s*<= 1'b0;", master)
        rise = re.search(r"2'd1: begin\s*sck\s*<= 1'b1;", master)
        fall = re.search(r"default: begin\s*sck\s*<= 1'b0;\s*frame", master)
        assert phases and rise and fall, "the master's phase contract moved"


# ---------------------------------------------------------------------------
# U-SP4 (L2): the SCK clock is defined where only the clock arc reaches, and nothing
# shortens or hides its latency instead
# ---------------------------------------------------------------------------

#: Commands the text may never carry on the kernel's clocks, pins, nets or gate cells: each
#: shortens a clock's latency by declaration, removes an arc, or hides a path.
FORBIDDEN_COMMANDS = ("set_clock_latency", "set_disable_timing", "set_case_analysis",
                      "set_clock_sense", "set_propagated_clock", "create_clock",
                      "set_false_path", "set_clock_groups", "set_min_delay")
#: Revision 3's own numbers: divide by 4, the hold multicycle 3, the factor 2.0.
ALLOWED_LITERALS = {"4", "3", "2.0"}
_LITERAL = re.compile(r"(?<![\w\[./-])-?\d+(?:\.\d+)?(?![\w\]])")


def _rule_violations(text: str, nc: int) -> list[str]:
    """Every way ``text`` departs from the U-SP4 constraint rules at ``nc`` cores, by name."""
    lines = _code(text)
    reasons: list[str] = []
    for line in lines:
        command = line.split()[0]
        if command in FORBIDDEN_COMMANDS:
            reasons.append(f"forbidden command {command}")
        if "sck_reg/Q" in line and "sck_reg/D" in line:
            reasons.append(f"an exception on the SCK register's feedback: {command}")
        if command != "create_generated_clock" and "sck_reg/D" in line:
            reasons.append(f"an exception on the SCK register's feedback: {command}")
        for literal in _LITERAL.findall(line):
            if literal not in ALLOWED_LITERALS:
                reasons.append(f"literal value {literal}")
        for core in re.findall(r"core_gen\[(\d+)\]", line):
            if int(core) >= nc:
                reasons.append(f"core {core} named beyond NC = {nc}")
        if re.search(r"odin_[0-9]|level0|ulp/|-hierarchical", line):
            reasons.append(f"names a unit or a level above the kernel top: {command}")
    if lines.count(CORE_CLOCK_LINE) != 1:
        reasons.append(f"the gated core clock's buffer request appears {lines.count(CORE_CLOCK_LINE)} times")
    rule_names = ("generated clock at the gate's clock input", "hold multicycle into SCK",
                  "slave-to-core data-path bound", "SCK buffer request", "ASYNC_REG")
    expected = {CORE_CLOCK_LINE}
    for c in range(nc):
        for name, line in zip(rule_names, _expected_core_lines(c)):
            expected.add(line)
            if lines.count(line) != 1:
                reasons.append(f"core {c}: {name} appears {lines.count(line)} times")
    for line in lines:
        if line not in expected and not line.split()[0] in FORBIDDEN_COMMANDS:
            reasons.append(f"a line outside the rules: {line.split()[0]}")
    return reasons


def _text(tmp_path: Path, cores: str) -> str:
    root, result = _build(tmp_path, ODIN_KERNEL_CORES=cores)
    assert result.returncode == 0, result.stdout + result.stderr
    return (root / "build" / "hacc" / _leaf(cores) / XDC_NAME).read_text()


def _replace(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _drop(text: str, line: str) -> str:
    return _replace(text, line + "\n", "")


def _add(text: str, line: str) -> str:
    return text + line + "\n"


_Q0, _D0, _C0 = (f"[get_pins {{kernel_i/core_gen[0].spi_i/sck_reg/{p}}}]" for p in "QDC")
_GATE0 = ("[get_cells -filter {PRIMITIVE_SUBGROUP == LUT} "
          f"[all_fanin -flat -only_cells {_C0}]]")
_REV3_SOURCE_LINE = f"create_generated_clock -source {_C0} -divide_by 4 {_Q0}"
#: The gated net's leaf driver from sck_reg/C, revision 3's buffer-request query: at a
#: build's link it is the gate LUT's output, in a routed checkpoint the inserted BUFGCE's.
_LEAF_DRIVER_SOURCE_LINE = (
    "create_generated_clock -source [get_pins -leaf -filter {DIRECTION == OUT} -of_objects "
    f"[get_nets -of_objects {_C0}]] -divide_by 4 {_Q0}")

#: name -> (edit of the NC 2 text, a reason the rules must give)
MUTANTS = {
    "revision 3's source, the SCK register's clock pin": (
        lambda t: _replace(t, _expected_core_lines(0)[0], _REV3_SOURCE_LINE),
        "core 0: generated clock at the gate's clock input appears 0 times"),
    "the source at the gated net's leaf driver": (
        lambda t: _replace(t, _expected_core_lines(1)[0],
                           _LEAF_DRIVER_SOURCE_LINE.replace("core_gen[0]", "core_gen[1]")),
        "core 1: generated clock at the gate's clock input appears 0 times"),
    "no generated clock on core 1": (
        lambda t: _drop(t, _expected_core_lines(1)[0]),
        "core 1: generated clock at the gate's clock input appears 0 times"),
    "no hold multicycle on core 0": (
        lambda t: _drop(t, _expected_core_lines(0)[1]),
        "core 0: hold multicycle into SCK appears 0 times"),
    "no slave bound on core 1": (
        lambda t: _drop(t, _expected_core_lines(1)[2]),
        "core 1: slave-to-core data-path bound appears 0 times"),
    "no SCK buffer request on core 0": (
        lambda t: _drop(t, _expected_core_lines(0)[3]),
        "core 0: SCK buffer request appears 0 times"),
    "no ASYNC_REG on core 1": (
        lambda t: _drop(t, _expected_core_lines(1)[4]),
        "core 1: ASYNC_REG appears 0 times"),
    "no core-clock buffer request": (
        lambda t: _drop(t, CORE_CLOCK_LINE),
        "the gated core clock's buffer request appears 0 times"),
    "the feedback false-pathed": (
        lambda t: _add(t, f"set_false_path -from {_Q0} -to {_D0}"),
        "an exception on the SCK register's feedback: set_false_path"),
    "the feedback bounded data path only": (
        lambda t: _add(t, f"set_max_delay -datapath_only -from {_Q0} -to {_D0} "
                          f"[expr {{2.0 * [get_property PERIOD [get_clocks -of_objects {_C0}]]}}]"),
        "an exception on the SCK register's feedback: set_max_delay"),
    "the feedback given a setup multicycle": (
        lambda t: _add(t, f"set_multicycle_path -setup -start 3 -from {_Q0} -to {_D0}"),
        "an exception on the SCK register's feedback: set_multicycle_path"),
    "a typed source latency on SCK": (
        lambda t: _add(t, f"set_clock_latency -source 0 [get_clocks -of_objects {_Q0}]"),
        "forbidden command set_clock_latency"),
    "the gate's side arcs disabled": (
        lambda t: _add(t, f"set_disable_timing -from I1 -to O {_GATE0}"),
        "forbidden command set_disable_timing"),
    "the gate's enable held by case analysis": (
        lambda t: _add(t, f"set_case_analysis 1 [get_pins -filter {{DIRECTION == IN}} -of_objects {_GATE0}]"),
        "forbidden command set_case_analysis"),
    "a clock sense on the gate": (
        lambda t: _add(t, f"set_clock_sense -positive [get_pins -filter {{DIRECTION == OUT}} -of_objects {_GATE0}]"),
        "forbidden command set_clock_sense"),
    "a propagated-clock declaration": (
        lambda t: _add(t, f"set_propagated_clock [get_clocks -of_objects {_Q0}]"),
        "forbidden command set_propagated_clock"),
    "a primary clock on SCK": (
        lambda t: _add(t, f"create_clock -period 40 {_Q0}"),
        "forbidden command create_clock"),
    "a literal time value": (
        lambda t: t.replace("2.0 *", "2.5 *", 1),
        "literal value 2.5"),
    "a unit named": (
        lambda t: t.replace("{kernel_i/core_gen[1].spi_i/sck_reg/Q}",
                            "{level0_i/ulp/odin_0/inst/kernel_i/core_gen[1].spi_i/sck_reg/Q}", 1),
        "names a unit or a level above the kernel top: create_generated_clock"),
    "a core beyond NC": (
        lambda t: _add(t, _expected_core_lines(2)[3]),
        "core 2 named beyond NC = 2"),
}


class TestTheSckClockIsDefinedAtTheGatesClockInput:
    """U-SP4 L2: the text keeps every rule of revision 3 per core, defines each SCK clock at
    the core-clock gate's clock input, and carries no form that would shorten or hide the
    SCK clock's latency (survey build lock revision 2, follow_up_unit.must_keep)."""

    @pytest.mark.parametrize("cores", ["1", "2", "4", "7"])
    def test_the_text_keeps_every_rule_at_every_core_count(self, tmp_path, cores):
        assert _rule_violations(_text(tmp_path, cores), int(cores)) == []

    def test_every_core_takes_the_gate_from_its_own_sck_register(self, tmp_path):
        lines = _code(_text(tmp_path, "4"))
        generated = [line for line in lines if line.startswith("create_generated_clock")]
        assert len(generated) == 4
        for c, line in enumerate(generated):
            assert line == (f"create_generated_clock -source {_gate_clock_input(c)} "
                            f"-divide_by 4 {_sck(c, 'Q')}")
            # never the register's own clock pin, never a net's leaf driver
            assert f"-source {_sck(c, 'C')}" not in line and "-leaf" not in line

    @pytest.mark.parametrize("name", sorted(MUTANTS))
    def test_a_mutant_of_the_text_fails_by_name(self, tmp_path, name):
        text = _text(tmp_path, "2")
        edit, reason = MUTANTS[name]
        mutant = edit(text)
        assert mutant != text
        assert reason in _rule_violations(mutant, 2), _rule_violations(mutant, 2)


class TestTheConstraintNamesAreTheRtls:

    def test_the_kernel_top_instantiates_the_kernel_as_kernel_i(self):
        top = (REPO / "hw" / "fpga" / "kernel" / "odin_fpga_kernel_top.v").read_text()
        assert re.search(r"odin_fpga_kernel\s+#\([^;]*?\)\s*kernel_i\s*\(", top, re.S)

    def test_each_core_has_its_spi_master_and_its_vendored_core(self):
        body = (REPO / "hw" / "fpga" / "kernel" / "odin_fpga_kernel.v").read_text()
        assert "begin : core_gen" in body
        assert re.search(r"odin_spi_master\s+spi_i\s*\(", body)
        assert re.search(r"ODIN\s+#\([^;]*?\)\s*dut\s*\(", body, re.S)
        assert re.search(r"\.SCK\s*\(sck_w\)", body) and re.search(r"\.sck\s*\(sck_w\)", body)
        # One gated core clock clocks every core's SPI master and vendored core alike.
        assert re.search(r"wire core_clk = clk & \(core_en_q \| rst\);", body)
        assert re.search(r"odin_spi_master spi_i \(\s*\.clk \(core_clk\)", body)
        assert re.search(r"\.CLK\s*\(core_clk\)", body)

    def test_the_vendored_core_names_its_slave_controller_and_scheduler(self):
        odin = (REPO / "hw" / "vendor" / "odin" / "src" / "ODIN.v").read_text(errors="replace")
        for instance in ("spi_slave_0", "controller_0", "scheduler_0"):
            assert re.search(rf"\)\s*{instance}\s*\(", odin), instance

    def test_every_synchronizer_register_exists(self):
        src = REPO / "hw" / "vendor" / "odin" / "src"
        texts = {"controller_0": (src / "controller.v").read_text(errors="replace"),
                 "scheduler_0": (src / "scheduler.v").read_text(errors="replace")}
        for reg in SYNC_REGS:
            instance, name = reg.split("/")
            signal = name[: -len("_reg")]
            assert re.search(rf"\b{signal}\s*<=", texts[instance]), reg

    def test_the_slave_is_the_only_logic_on_sck(self):
        """SCK clocks the slave and nothing else, so the domain is the slave's cells."""
        slave = (REPO / "hw" / "vendor" / "odin" / "src" / "spi_slave.v").read_text(
            errors="replace")
        assert re.search(r"always @\(negedge SCK, posedge RST_async\)", slave)
        assert re.search(r"always @\(posedge SCK\)", slave)
        for name in ("controller.v", "scheduler.v", "neuron_core.v", "synaptic_core.v",
                     "aer_out.v"):
            text = (REPO / "hw" / "vendor" / "odin" / "src" / name).read_text(errors="replace")
            assert not re.search(r"\bSCK\b", text), name


# ---------------------------------------------------------------------------
# The packaging carries it, and the build checks that it did
# ---------------------------------------------------------------------------


class TestThePackagedCoreCarriesTheConstraint:

    def test_the_constraint_is_added_for_implementation_late(self, tmp_path):
        root, result = _build(tmp_path, ODIN_KERNEL_CORES="2")
        assert result.returncode == 0, result.stdout + result.stderr
        build = root / "build" / "hacc" / _leaf("2")
        tcl = (build / "pack_kernel.tcl").read_text()
        xdc = str((build / XDC_NAME).resolve())
        assert f'add_files -fileset constrs_1 -norecurse "{xdc}"' in tcl
        for prop in ("USED_IN_SYNTHESIS false", "USED_IN_IMPLEMENTATION true",
                     "PROCESSING_ORDER LATE"):
            assert f"set_property {prop} $spi_xdc" in tcl, prop
        # Added before the core is packaged, or the packaged core never sees it.
        assert tcl.index("add_files -fileset constrs_1") < tcl.index("ipx::package_project")
        assert tcl.index("PROCESSING_ORDER LATE") < tcl.index("ipx::package_project")
        assert "ODIN_PACKAGED_XDC" in tcl
        assert tcl.index("ODIN_PACKAGED_XDC") < tcl.index("ipx::save_core")
        assert "[hacc-build] packaged : spi xdc for implementation, late" in result.stdout

    @pytest.mark.parametrize("answer", ["none", "synthesis implementation late",
                                        "implementation normal"])
    def test_a_core_packaged_without_it_refuses_before_the_link(self, tmp_path, answer):
        root, result = _build(tmp_path, ODIN_KERNEL_CORES="1",
                              ODIN_STUB_PACKAGED_XDC=answer)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "REFUSING" in result.stderr and XDC_NAME in result.stderr
        assert not (tmp_path / "vxx.trace").exists()

    def test_a_fabric_without_the_vendored_slave_gets_none(self, tmp_path):
        """The generated core has a synchronous write port and no SCK: nothing to time."""
        root, result = _build(tmp_path, ODIN_CHIP="odin_wide_1024x256_mb16")
        assert result.returncode == 0, result.stdout + result.stderr
        build = root / "build" / "hacc" / _leaf("1", "1", "_odin_wide_1024x256_mb16")
        assert not (build / XDC_NAME).exists()
        tcl = (build / "pack_kernel.tcl").read_text()
        assert "constrs_1" not in tcl and "ODIN_PACKAGED_XDC" not in tcl
        assert "[hacc-build] spi xdc  : none (no SPI clock domain in this fabric)" in result.stdout
        assert (tmp_path / "vxx.trace").exists()

    def test_the_stub_vivado_answers_what_the_tcl_asked(self, tmp_path):
        """The stub plays a faithful packaging, so the refusal tests mean something."""
        root, result = _build(tmp_path, ODIN_KERNEL_CORES="1")
        assert result.returncode == 0, result.stdout + result.stderr
        log = (root / "build" / "hacc" / _leaf("1") / "pack_kernel.log").read_text()
        assert f"ODIN_PACKAGED_XDC {XDC_NAME} implementation late xilinx_implementation" in log
