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
    register's own clock), a hold multicycle of 3 core cycles into the domain
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


def _expected_core_lines(c: int) -> list:
    master = f"[get_clocks -of_objects {_sck(c, 'C')}]"
    sck = f"[get_clocks -of_objects {_sck(c, 'Q')}]"
    slave = f"[get_cells -filter {{IS_SEQUENTIAL}} {{kernel_i/core_gen[{c}].dut/spi_slave_0/*}}]"
    syncs = " ".join(f"kernel_i/core_gen[{c}].dut/{reg}" for reg in SYNC_REGS)
    return [
        f"create_generated_clock -source {_sck(c, 'C')} -divide_by 4 {_sck(c, 'Q')}",
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

    @pytest.mark.parametrize("cores", ["1", "2", "4"])
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
