"""[ODIN multi-core, Stage 1] NC cores per compute unit: the count is data, end to end.

The kernel instantiates NC ODIN cores in its ``core_gen`` loop, each with its own
SPI master and AER-in bridge and one shared capture (``hw/fpga/kernel/
odin_fpga_kernel.v``). Stage 1 builds NC > 1 at the stock geometry; spikes
between cores cross at segment boundaries through the host. The claims gated
here, off-cluster (stub Vivado and v++):

  * ``scripts/hacc/chips.sh`` reads NC from ``ODIN_KERNEL_CORES`` (default 1),
    refuses a value that is not an integer from 1 to the 256 cores the kernel
    can address (``active_core[7:0]``), and names the build directory leaf
    ``<target>_nc<NC><chip suffix>`` -- the one place a script learns it;
  * ``build_xclbn.sh`` packages the kernel with its ``NC`` parameter set to that
    value (user and HDL parameter of the packaged core), refuses unless the
    packaging log confirms both, and links into ``<target>_nc<NC>`` (with the
    compute-unit suffix after it); NC = 1 lands where it always did;
  * ``odin_build.sbatch`` reports the artefacts of that same directory;
  * no build or package script spells a build directory with a literal ``nc1``.
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
CHIPS_SH = HACC / "chips.sh"
STUBS = HACC / "selftest" / "stubs" / "vitis_bin"

#: Every variable a build reads from the environment, cleared per test.
BUILD_ENV = ("ODIN_COMPUTE_UNITS", "ODIN_CHIP", "ODIN_CU_SLRS", "ODIN_CARD",
             "ODIN_KERNEL_CORES", "ODIN_LOG", "ODIN_STUB_PACKAGED_NC")


def _bash(script: str, env=None, cwd=REPO) -> subprocess.CompletedProcess:
    merged = {k: v for k, v in os.environ.items() if k not in BUILD_ENV}
    merged.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=str(cwd), env=merged)


# ---------------------------------------------------------------------------
# chips.sh: the count and the directory it names
# ---------------------------------------------------------------------------


class TestTheCoreCountIsData:

    def test_the_count_defaults_to_one(self):
        result = _bash(f'source "{CHIPS_SH}"; printf "%s" "$(odin_kernel_cores)"')
        assert result.returncode == 0, result.stderr
        assert result.stdout == "1"

    @pytest.mark.parametrize("value", ["2", "3", "4", "256"])
    def test_a_positive_count_is_read_from_the_environment(self, value):
        result = _bash(f'source "{CHIPS_SH}"; printf "%s" "$(odin_kernel_cores)"',
                       env={"ODIN_KERNEL_CORES": value})
        assert result.returncode == 0, result.stderr
        assert result.stdout == value

    @pytest.mark.parametrize("value", ["0", "x", "-2", "1.5", "257", "04x"])
    def test_a_count_the_kernel_cannot_address_is_refused(self, value):
        result = _bash(f'source "{CHIPS_SH}"; odin_kernel_cores',
                       env={"ODIN_KERNEL_CORES": value})
        assert result.returncode != 0
        assert "ODIN_KERNEL_CORES" in result.stderr
        assert "256" in result.stderr

    def test_the_limit_is_the_kernels_own_core_index_width(self):
        """256 is not a choice: the sequencer and the capture index cores with 8 bits."""
        kernel = (REPO / "hw" / "fpga" / "kernel" / "odin_fpga_kernel.v").read_text()
        assert "active_core[7:0]" in kernel and "cap_core[7:0]" in kernel
        result = _bash(f'source "{CHIPS_SH}"; printf "%s" "${{ODIN_KERNEL_CORES_MAX}}"')
        assert result.stdout == str(1 << 8)

    @pytest.mark.parametrize("cores, chip, leaf", [
        (None, None, "hw_nc1"),
        ("2", None, "hw_nc2"),
        ("4", "odin_wide_1024x256_mb16", "hw_nc4_odin_wide_1024x256_mb16"),
    ])
    def test_the_build_leaf_carries_the_count_and_the_chip(self, cores, chip, leaf):
        env = {}
        if cores is not None:
            env["ODIN_KERNEL_CORES"] = cores
        if chip is not None:
            env["ODIN_CHIP"] = chip
        result = _bash(f'source "{CHIPS_SH}"; printf "%s" "$(odin_build_leaf hw)"', env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout == leaf

    def test_a_named_chip_overrides_the_selected_one(self):
        result = _bash(f'source "{CHIPS_SH}"; '
                       f'printf "%s" "$(odin_build_leaf hw_emu odin_wide_1024x256_mb16)"',
                       env={"ODIN_KERNEL_CORES": "3"})
        assert result.stdout == "hw_emu_nc3_odin_wide_1024x256_mb16"

    def test_a_refused_count_names_no_directory(self):
        result = _bash(f'source "{CHIPS_SH}"; odin_build_leaf hw',
                       env={"ODIN_KERNEL_CORES": "0"})
        assert result.returncode != 0
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# build_xclbn.sh and odin_build.sbatch, against stub Vivado and v++
# ---------------------------------------------------------------------------


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


def _build(root: Path, env: dict, **extra: str) -> subprocess.CompletedProcess:
    run_env = {k: v for k, v in os.environ.items() if k not in BUILD_ENV}
    run_env.update(env)
    run_env.update(extra)
    return subprocess.run(["bash", "scripts/hacc/build_xclbn.sh", "hw"], capture_output=True,
                          text=True, cwd=str(root), env=run_env)


def _argv(tmp_path: Path) -> str:
    return (tmp_path / "vxx.trace").read_text().splitlines()[-1]


class TestTheBuildPackagesNCores:

    @pytest.mark.parametrize("cores", ["2", "4"])
    def test_the_packaged_core_is_given_the_count(self, tmp_path, cores):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES=cores)
        assert result.returncode == 0, result.stdout + result.stderr
        tcl = (root / "build" / "hacc" / f"hw_nc{cores}" / "pack_kernel.tcl").read_text()
        assert f"set kernel_cores {cores}\n" in tcl
        # Both halves of the packaged parameter: what an instance is configured
        # with (the user parameter) and what elaborates the HDL (the model one).
        assert "ipx::get_user_parameters NC" in tcl
        assert "ipx::get_hdl_parameters NC" in tcl
        assert "set_property value $kernel_cores" in tcl
        assert "ODIN_PACKAGED_NC" in tcl
        # The parameter is set BEFORE the core is saved, or the saved IP keeps 1.
        assert tcl.index("ipx::get_user_parameters NC") < tcl.index("ipx::save_core")
        assert f"[odin-build] cores    : {cores}" in result.stdout

    @pytest.mark.parametrize("cores", ["2", "4"])
    def test_n_cores_link_into_their_own_directory(self, tmp_path, cores):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES=cores)
        assert result.returncode == 0, result.stdout + result.stderr
        argv = _argv(tmp_path)
        assert "--config scripts/hacc/odin_u55c.cfg" in argv
        assert f"-o build/hacc/hw_nc{cores}/odin_fpga_hw.xclbin" in argv
        assert not (root / "build" / "hacc" / "hw_nc1").exists()

    def test_one_core_is_packaged_explicitly_and_lands_where_it_always_did(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "-o build/hacc/hw_nc1/odin_fpga_hw.xclbin" in _argv(tmp_path)
        tcl = (root / "build" / "hacc" / "hw_nc1" / "pack_kernel.tcl").read_text()
        assert "set kernel_cores 1\n" in tcl

    def test_the_count_and_the_unit_count_compose(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES="2",
                        ODIN_COMPUTE_UNITS="3")
        assert result.returncode == 0, result.stdout + result.stderr
        argv = _argv(tmp_path)
        assert "--config build/hacc/hw_nc2_cu3/link_cu3.cfg" in argv
        assert "-o build/hacc/hw_nc2_cu3/odin_fpga_hw.xclbin" in argv

    def test_a_packaging_that_ignored_the_count_refuses_before_the_link(self, tmp_path):
        """The packaging log must CONFIRM the count, or the link is never started."""
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES="4",
                        ODIN_STUB_PACKAGED_NC="1 1")
        assert result.returncode == 2, result.stdout + result.stderr
        assert "REFUSING" in result.stderr and "NC" in result.stderr
        assert not (tmp_path / "vxx.trace").exists()

    def test_an_unaddressable_count_refuses_before_any_tool_runs(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES="300")
        assert result.returncode == 2
        assert "ODIN_KERNEL_CORES" in result.stderr
        assert not (tmp_path / "vxx.trace").exists()
        assert not (root / "build").exists()

    def test_the_refusal_of_nc_above_one_is_gone(self):
        text = (HACC / "build_xclbn.sh").read_text()
        assert "plumbs NC=1 only" not in text


class TestTheBuildJobReportsTheSameDirectory:

    def test_the_build_job_lists_the_nc_artefacts(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        env.update({"ODIN_PKG": str(root), "ODIN_TARGET": "hw", "ODIN_KERNEL_CORES": "4"})
        run_env = {k: v for k, v in os.environ.items() if k not in BUILD_ENV}
        run_env.update(env)
        result = subprocess.run(["bash", str(root / "scripts" / "hacc" / "odin_build.sbatch")],
                                capture_output=True, text=True, cwd=str(tmp_path), env=run_env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"artefacts under {root}/build/hacc/hw_nc4/" in result.stdout
        assert "[build] cores per unit 4" in result.stdout


# ---------------------------------------------------------------------------
# No script spells NC = 1 into a path
# ---------------------------------------------------------------------------

#: Every shell script of the build and the package, and the package's job files.
SCRIPTS = sorted(
    list(HACC.glob("*.sh"))
    + list((HACC / "package").glob("*.sh"))
    + list((HACC / "package" / "scripts").glob("*.sh"))
    + list((HACC / "package" / "sbatch").glob("*.sbatch")))


def _code_lines(path: Path):
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        yield number, line


class TestNoScriptHardWiresOneCore:

    def test_the_scripts_are_found(self):
        names = {path.name for path in SCRIPTS}
        assert {"build_xclbn.sh", "chips.sh", "run_all.sh", "collect_results.sh",
                "chip_cache.sh", "mine_checkpoint.sh", "odin_build.sbatch"} <= names

    @pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
    def test_no_build_directory_is_spelled_with_a_literal_nc1(self, path):
        offending = [f"{path.name}:{number}: {line.strip()}"
                     for number, line in _code_lines(path)
                     if re.search(r"_nc1(?![0-9])", line)]
        assert not offending, "\n".join(offending)


class TestThePackageScriptsFollowTheCount:

    def test_the_chip_cache_finds_the_nc_build_directory(self, tmp_path):
        """chip_cache.sh's xclbin path and key follow ODIN_KERNEL_CORES, not the RTL default."""
        package = REPO / "scripts" / "hacc" / "package" / "scripts" / "chip_cache.sh"
        text = package.read_text()
        assert "odin_build_leaf" in text
        assert re.search(r'local nc="\$\{7:-\$\(odin_kernel_cores\)\}"', text), (
            "the cache key's nc field must be the build's NC")

    @pytest.mark.parametrize("relative", [
        "scripts/hacc/package/run_all.sh", "scripts/hacc/package/collect_results.sh",
        "scripts/hacc/mine_checkpoint.sh"])
    def test_every_script_that_names_a_build_directory_asks_chips_sh(self, relative):
        assert "odin_build_leaf" in (REPO / relative).read_text()


class TestTheBoardJobCarriesTheCount:

    def test_the_board_job_documents_the_declared_cores(self):
        text = (HACC / "package" / "sbatch" / "odin_board.sbatch").read_text()
        assert "ODIN_KERNEL_CORES=NC declares the cores" in text

    def test_the_fixture_run_forwards_the_placement_mode(self):
        text = (HACC / "package" / "sbatch" / "odin_board.sbatch").read_text()
        assert '--placements "${ODIN_PLACEMENTS:-identity}"' in text


class TestTheBuildRecordsItsCoreCountWithItsIdentity:
    """[targeted pass] NC travels WITH the bitstream, so the host can check a declaration.

    The host used to trust ``--declare-cores`` or ``ODIN_KERNEL_CORES`` alone:
    nothing tied a loaded xclbin to the NC it was built with. The build now
    writes ``<xclbin>.identity`` beside every xclbin it links: the xclbin's
    sha256, its UUID (read off the link's own ``.info``), the NC it packaged,
    the chip and the unit count.
    """

    @staticmethod
    def _record(path: Path) -> dict:
        return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)

    @pytest.mark.parametrize("cores", ["1", "2", "4"])
    def test_the_xclbin_carries_an_identity_record_with_its_nc(self, tmp_path, cores):
        import hashlib

        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES=cores,
                        ODIN_STUB_XCLBIN_UUID="5770b000-0000-4000-8000-0000000000aa")
        assert result.returncode == 0, result.stdout + result.stderr
        xclbin = root / "build" / "hacc" / f"hw_nc{cores}" / "odin_fpga_hw.xclbin"
        record = self._record(xclbin.parent / (xclbin.name + ".identity"))
        assert record["kernel_cores"] == cores
        assert record["xclbin_sha256"] == hashlib.sha256(xclbin.read_bytes()).hexdigest()
        assert record["xclbin_uuid"] == "5770b000-0000-4000-8000-0000000000aa"
        assert record["chip_config"] == "odin_stock_256x256"
        assert record["compute_units"] == "1"
        assert record["target"] == "hw"
        assert f"[odin-build] identity : NC {cores}" in result.stdout

    def test_the_units_and_the_cores_both_reach_the_record(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        result = _build(root, _stub_toolchain(tmp_path), ODIN_KERNEL_CORES="2",
                        ODIN_COMPUTE_UNITS="3")
        assert result.returncode == 0, result.stdout + result.stderr
        record = self._record(
            root / "build" / "hacc" / "hw_nc2_cu3" / "odin_fpga_hw.xclbin.identity")
        assert record["kernel_cores"] == "2" and record["compute_units"] == "3"
