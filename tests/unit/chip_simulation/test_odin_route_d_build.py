"""[ODIN route D] N compute units in the link: the count is data, the banks follow it.

Route D builds N compute units of the SAME kernel into one xclbin, each on its
own memory bank, so the card-boundary workload increment carries N kernels. The
claims gated here, off-cluster (stub Vivado and v++):

  * ``scripts/hacc/gen_link_cfg.py`` derives the link's ``[connectivity]`` from
    the card's own v++ config and N alone: at N = 1 it is the identity (the
    archived config, byte for byte); at N > 1 it names N units after the base
    config's unit and puts unit k on bank (first bank + k); every other line of
    the config is kept; a count past the card's declared banks is refused;
  * ``scripts/hacc/cards.sh`` declares each card's bank count beside its memory,
    reads N from ``ODIN_COMPUTE_UNITS`` (default 1) and names the build
    directory suffix (none at N = 1, so the archived directories do not move);
  * ``build_xclbn.sh`` hands v++ the card's own config at N = 1 and the
    generated one at N > 1, into a ``_cu<N>`` directory, for either chip;
  * ``odin_build.sbatch`` reports the artefacts of that same directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HACC = REPO / "scripts" / "hacc"
GEN = HACC / "gen_link_cfg.py"
CARDS_SH = HACC / "cards.sh"
U55C_CFG = HACC / "odin_u55c.cfg"
U250_CFG = HACC / "odin_u250.cfg"
STUBS = HACC / "selftest" / "stubs" / "vitis_bin"


def _gen_module():
    spec = importlib.util.spec_from_file_location("gen_link_cfg_under_test", GEN)
    assert spec is not None and spec.loader is not None, f"{GEN} is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bash(script: str, env=None, cwd=REPO) -> subprocess.CompletedProcess:
    merged = dict(os.environ)
    merged.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=str(cwd), env=merged)


# ---------------------------------------------------------------------------
# The link configuration
# ---------------------------------------------------------------------------


class TestTheLinkConfigIsDerivedFromNAlone:

    def test_one_unit_is_the_archived_config_byte_for_byte(self):
        gen = _gen_module()
        base = U55C_CFG.read_text()
        assert gen.link_config(base, units=1, banks=32) == base

    def test_ten_units_are_named_after_the_base_unit_on_ten_banks(self):
        gen = _gen_module()
        text = gen.link_config(U55C_CFG.read_text(), units=10, banks=32)
        lines = [line.strip() for line in text.splitlines()]
        names = [f"odin_{k}" for k in range(10)]
        assert f"nk=odin_fpga_kernel_top:10:{'.'.join(names)}" in lines
        for k, name in enumerate(names):
            assert f"sp={name}.m_axi_gmem:HBM[{k}]" in lines
        assert sum(1 for line in lines if line.startswith("nk=")) == 1
        assert sum(1 for line in lines if line.startswith("sp=")) == 10
        assert gen.compute_unit_names(U55C_CFG.read_text(), 10) == names

    def test_every_line_outside_the_connectivity_lines_is_kept(self):
        gen = _gen_module()
        base = U55C_CFG.read_text()
        text = gen.link_config(base, units=10, banks=32)
        kept = [line for line in base.splitlines()
                if not line.startswith(("nk=", "sp="))]
        produced = text.splitlines()
        position = 0
        for line in kept:
            position = produced.index(line, position) + 1
        assert "[vivado]" in text and "defaultFreqHz=100000000" in text

    def test_slr_assignments_are_data_one_per_unit(self):
        gen = _gen_module()
        slrs = ["SLR0"] * 5 + ["SLR1"] * 5
        text = gen.link_config(U55C_CFG.read_text(), units=10, banks=32, slrs=slrs)
        lines = [line.strip() for line in text.splitlines()]
        for k, slr in enumerate(slrs):
            assert f"slr=odin_{k}:{slr}" in lines

    @pytest.mark.parametrize("units, banks, slrs, needle", [
        (0, 32, None, "at least one"),
        (33, 32, None, "declares 32"),
        (3, 32, ["SLR0", "SLR1"], "3 compute units"),
        (2, 32, ["SLR0", "SLR9x"], "not an SLR"),
    ])
    def test_impossible_requests_are_refused_by_name(self, units, banks, slrs, needle):
        gen = _gen_module()
        with pytest.raises(gen.LinkConfigRefusal, match=needle):
            gen.link_config(U55C_CFG.read_text(), units=units, banks=banks, slrs=slrs)

    def test_the_u250_config_keeps_its_ddr_family_and_its_four_banks(self):
        gen = _gen_module()
        base = U250_CFG.read_text()
        assert gen.link_config(base, units=1, banks=4) == base
        text = gen.link_config(base, units=4, banks=4)
        assert "sp=odin_3.m_axi_gmem:DDR[3]" in [line.strip() for line in text.splitlines()]
        with pytest.raises(gen.LinkConfigRefusal, match="declares 4"):
            gen.link_config(base, units=5, banks=4)

    def test_the_command_line_writes_the_generated_config(self, tmp_path):
        out = tmp_path / "link.cfg"
        result = subprocess.run(
            [sys.executable, str(GEN), "--base", str(U55C_CFG), "--compute-units", "3",
             "--memory-banks", "32", "--slrs", "SLR0,SLR0,SLR1", "--output", str(out)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        text = out.read_text()
        assert "nk=odin_fpga_kernel_top:3:odin_0.odin_1.odin_2" in text
        assert "slr=odin_2:SLR1" in text
        refused = subprocess.run(
            [sys.executable, str(GEN), "--base", str(U55C_CFG), "--compute-units", "40",
             "--memory-banks", "32", "--output", str(tmp_path / "no.cfg")],
            capture_output=True, text=True)
        assert refused.returncode == 2 and "REFUSING" in refused.stderr
        assert not (tmp_path / "no.cfg").exists()


# ---------------------------------------------------------------------------
# The card profile and the compute-unit count
# ---------------------------------------------------------------------------


class TestTheCardDeclaresItsBanks:

    @pytest.mark.parametrize("card, banks", [("u55c", "32"), ("u250", "4")])
    def test_every_card_declares_its_memory_banks(self, card, banks):
        result = _bash(f'source "{CARDS_SH}"; printf "%s" "$(odin_card_field memory_banks {card})"')
        assert result.returncode == 0, result.stderr
        assert result.stdout == banks

    def test_the_count_defaults_to_one_and_the_suffix_to_nothing(self):
        result = _bash(f'source "{CARDS_SH}"; unset ODIN_COMPUTE_UNITS; '
                       f'printf "%s|%s|" "$(odin_compute_units)" "$(odin_cu_suffix 1)"')
        assert result.returncode == 0, result.stderr
        assert result.stdout == "1||"

    def test_ten_units_name_their_own_directory(self):
        result = _bash(f'source "{CARDS_SH}"; printf "%s|%s" "$(odin_compute_units)" '
                       f'"$(odin_cu_suffix "$(odin_compute_units)")"',
                       env={"ODIN_COMPUTE_UNITS": "10"})
        assert result.returncode == 0, result.stderr
        assert result.stdout == "10|_cu10"

    @pytest.mark.parametrize("value", ["0", "x", "-3", "1.5", ""])
    def test_a_count_that_is_not_a_positive_integer_is_refused(self, value):
        result = _bash(f'source "{CARDS_SH}"; odin_compute_units',
                       env={"ODIN_COMPUTE_UNITS": value})
        if value == "":
            # Unset-or-empty is the default, one unit.
            assert result.returncode == 0 and result.stdout == "1"
            return
        assert result.returncode != 0
        assert "ODIN_COMPUTE_UNITS" in result.stderr


# ---------------------------------------------------------------------------
# The build script and the build job, against stub Vivado and v++
# ---------------------------------------------------------------------------


def _stage_build_tree(root: Path) -> Path:
    """The files build_xclbn.sh reads, laid out as a package lays them out."""
    hacc = root / "scripts" / "hacc"
    hacc.mkdir(parents=True)
    for name in ("build_xclbn.sh", "toolchain.sh", "cards.sh", "chips.sh",
                 "odin_u55c.cfg", "odin_u250.cfg", "gen_link_cfg.py"):
        shutil.copyfile(HACC / name, hacc / name)
    shutil.copyfile(REPO / "scripts" / "hacc" / "package" / "sbatch" / "odin_build.sbatch",
                    hacc / "odin_build.sbatch")
    # A stand-in for the frozen kernel.xml step: the link under test reads none of it.
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


def _build(root: Path, env: dict, *, units: str | None, chip: str | None = None,
           slrs: str | None = None) -> subprocess.CompletedProcess:
    run_env = dict(env)
    run_env.pop("ODIN_COMPUTE_UNITS", None)
    if units is not None:
        run_env["ODIN_COMPUTE_UNITS"] = units
    if chip is not None:
        run_env["ODIN_CHIP"] = chip
    if slrs is not None:
        run_env["ODIN_CU_SLRS"] = slrs
    clean = {k: v for k, v in os.environ.items()
             if k not in ("ODIN_COMPUTE_UNITS", "ODIN_CHIP", "ODIN_CU_SLRS", "ODIN_CARD")}
    clean.update(run_env)
    return subprocess.run(["bash", "scripts/hacc/build_xclbn.sh", "hw"], capture_output=True,
                          text=True, cwd=str(root), env=clean)


def _argv(root: Path) -> str:
    return (root / "vxx.trace").read_text().splitlines()[-1]


class TestTheBuildScriptLinksNUnits:

    def test_one_unit_links_the_cards_own_config_where_it_always_did(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        result = _build(root, env, units=None)
        assert result.returncode == 0, result.stdout + result.stderr
        argv = _argv(tmp_path)
        assert "--config scripts/hacc/odin_u55c.cfg" in argv
        assert "-o build/hacc/hw_nc1/odin_fpga_hw.xclbin" in argv
        assert not list((root / "build" / "hacc").glob("*_cu*"))

    def test_ten_units_link_a_generated_config_in_their_own_directory(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        result = _build(root, env, units="10")
        assert result.returncode == 0, result.stdout + result.stderr
        argv = _argv(tmp_path)
        assert "--config build/hacc/hw_nc1_cu10/link_cu10.cfg" in argv
        assert "-o build/hacc/hw_nc1_cu10/odin_fpga_hw.xclbin" in argv
        generated = (root / "build" / "hacc" / "hw_nc1_cu10" / "link_cu10.cfg").read_text()
        assert generated == _gen_module().link_config(U55C_CFG.read_text(), units=10, banks=32)
        assert "[odin-build] compute units: 10" in result.stdout

    def test_the_widened_chip_and_its_slr_list_travel_with_the_count(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        slrs = ",".join(["SLR0"] * 5 + ["SLR1"] * 5)
        result = _build(root, env, units="10", chip="odin_wide_1024x256_mb16", slrs=slrs)
        assert result.returncode == 0, result.stdout + result.stderr
        build = "build/hacc/hw_nc1_odin_wide_1024x256_mb16_cu10"
        assert f"--config {build}/link_cu10.cfg" in _argv(tmp_path)
        generated = (root / build / "link_cu10.cfg").read_text()
        assert "slr=odin_9:SLR1" in generated and "slr=odin_0:SLR0" in generated

    def test_more_units_than_banks_refuses_before_any_tool_runs(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        result = _build(root, env, units="33")
        assert result.returncode == 2
        assert "REFUSING" in result.stderr
        assert not (tmp_path / "vxx.trace").exists()


class TestTheBuildJobReportsTheSameDirectory:

    def test_the_build_job_lists_the_cu10_artefacts(self, tmp_path):
        root = _stage_build_tree(tmp_path / "pkg")
        env = _stub_toolchain(tmp_path)
        env.update({"ODIN_PKG": str(root), "ODIN_TARGET": "hw", "ODIN_COMPUTE_UNITS": "10"})
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("ODIN_CHIP", "ODIN_CU_SLRS", "ODIN_CARD", "ODIN_LOG")}
        clean.update(env)
        result = subprocess.run(["bash", str(root / "scripts" / "hacc" / "odin_build.sbatch")],
                                capture_output=True, text=True, cwd=str(tmp_path), env=clean)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"artefacts under {root}/build/hacc/hw_nc1_cu10/" in result.stdout
        assert "[build] compute units 10" in result.stdout


# ---------------------------------------------------------------------------
# The route D overlay: the power overlay plus what the two rebuilds read
# ---------------------------------------------------------------------------

OVERLAY = HACC / "make_power_overlay.py"

#: The kernel.xml the archived September packages froze (sha256 of
#: scripts/hacc/kernel.xml in the stock install of odin_hacc_package and in the
#: widened package, read 2026-10-26). A rebuilt bitstream must be packaged from the same
#: register map, or its host and its kernel disagree.
ARCHIVED_KERNEL_XML_SHA256 = (
    "0411419c3599e42ad0e798e6475595c12f7e57e86c618e968397f040691cddf3")


def _overlay(tmp_path: Path, *flags: str) -> Path:
    result = subprocess.run(
        [sys.executable, str(OVERLAY), "--stage-only", "--out", str(tmp_path), *flags],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / "odin_power_overlay"


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTheRouteDOverlayCarriesTheBuild:

    def test_without_the_flag_the_power_overlay_is_unchanged(self, tmp_path):
        root = _overlay(tmp_path)
        names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        assert not any(name.startswith("hw/") for name in names)
        assert "scripts/hacc/build_xclbn.sh" not in names
        assert len(names) == 13

    def test_the_build_payload_is_the_repository_bytes(self, tmp_path):
        import json
        root = _overlay(tmp_path, "--build")
        for name in ("build_xclbn.sh", "toolchain.sh", "cards.sh", "chips.sh",
                     "odin_u55c.cfg", "odin_u250.cfg", "gen_link_cfg.py"):
            assert _sha(root / "scripts" / "hacc" / name) == _sha(HACC / name), name
        assert _sha(root / "scripts" / "hacc" / "odin_build.sbatch") == _sha(
            HACC / "package" / "sbatch" / "odin_build.sbatch")
        for chip in ("odin_stock_256x256", "odin_wide_1024x256_mb16"):
            listed = _bash(f'source scripts/hacc/chips.sh; ODIN_CHIP={chip} odin_chip_sources',
                           cwd=root)
            assert listed.returncode == 0, listed.stderr
            sources = [line for line in listed.stdout.splitlines() if line]
            assert sources, chip
            for relative in sources:
                assert _sha(root / relative) == _sha(REPO / relative), relative
        manifest = json.loads((root / "MANIFEST.json").read_text())
        assert manifest["build"] is True
        for relative, digest in manifest["files"].items():
            assert _sha(root / relative) == digest, relative

    def test_the_frozen_kernel_xml_is_the_archived_one_and_its_stand_in_verifies_it(
            self, tmp_path):
        root = _overlay(tmp_path, "--build")
        xml = root / "scripts" / "hacc" / "kernel.xml"
        assert _sha(xml) == ARCHIVED_KERNEL_XML_SHA256
        stand_in = (root / "scripts" / "hacc" / "gen_kernel_xml.py").read_text()
        assert "__ODIN_KERNEL_XML_SHA256__" not in stand_in
        out = tmp_path / "kernel_copy.xml"
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "hacc" / "gen_kernel_xml.py"),
             "--output", str(out), "--kernel", "odin_fpga_kernel_top"],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert _sha(out) == ARCHIVED_KERNEL_XML_SHA256
