"""Contracts for the MIMARSINAN_* environment-variable SSOT accessors."""

import os
from pathlib import Path

import pytest

from mimarsinan.common import env

SRC_ROOT = Path(env.__file__).resolve().parents[1]

EXACT_ONE_FLAGS = [
    (env.cuda_debug_enabled, "MIMARSINAN_CUDA_DEBUG"),
    (env.vram_probe_enabled, "MIMARSINAN_VRAM_PROBE"),
    (env.resource_debug_enabled, "MIMARSINAN_RESOURCE_DEBUG"),
    (env.nf_scm_parity_debug_enabled, "MIMARSINAN_NF_SCM_PARITY_DEBUG"),
    (env.ffcv_disabled, "MIMARSINAN_DISABLE_FFCV"),
    (env.loihi_quiet, "MIMARSINAN_LOIHI_QUIET"),
    (env.test_cuda_enabled, "MIMARSINAN_TEST_CUDA"),
    (env.mbh_ledger_enabled, "MIMARSINAN_MBH_LEDGER"),
    (env.mbh_lif_tanneal_enabled, "MIMARSINAN_MBH_LIF_TANNEAL"),
]
_FLAG_IDS = [var for _, var in EXACT_ONE_FLAGS]


class TestExactOneFlags:
    @pytest.mark.parametrize("accessor,var", EXACT_ONE_FLAGS, ids=_FLAG_IDS)
    def test_unset_is_false(self, monkeypatch, accessor, var):
        monkeypatch.delenv(var, raising=False)
        assert accessor() is False

    @pytest.mark.parametrize("accessor,var", EXACT_ONE_FLAGS, ids=_FLAG_IDS)
    def test_one_is_true(self, monkeypatch, accessor, var):
        monkeypatch.setenv(var, "1")
        assert accessor() is True

    @pytest.mark.parametrize("accessor,var", EXACT_ONE_FLAGS, ids=_FLAG_IDS)
    @pytest.mark.parametrize("value", ["0", "true", "yes", "", "2"])
    def test_anything_but_exact_one_is_false(self, monkeypatch, accessor, var, value):
        monkeypatch.setenv(var, value)
        assert accessor() is False

    def test_reads_environ_at_call_time(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_CUDA_DEBUG", raising=False)
        assert env.cuda_debug_enabled() is False
        monkeypatch.setenv("MIMARSINAN_CUDA_DEBUG", "1")
        assert env.cuda_debug_enabled() is True


class TestSetCudaDebug:
    def test_set_enables_flag(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_CUDA_DEBUG", raising=False)
        env.set_cuda_debug()
        assert os.environ["MIMARSINAN_CUDA_DEBUG"] == "1"
        assert env.cuda_debug_enabled() is True

    def test_clear_removes_variable(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_CUDA_DEBUG", "1")
        env.set_cuda_debug(False)
        assert "MIMARSINAN_CUDA_DEBUG" not in os.environ
        assert env.cuda_debug_enabled() is False

    def test_clear_when_unset_is_a_no_op(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_CUDA_DEBUG", raising=False)
        env.set_cuda_debug(False)
        assert "MIMARSINAN_CUDA_DEBUG" not in os.environ


class TestGuiNoBrowser:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_GUI_NO_BROWSER", raising=False)
        assert env.gui_no_browser() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes", " 1 ", "\ttrue\n"])
    def test_truthy_spellings_are_true(self, monkeypatch, value):
        monkeypatch.setenv("MIMARSINAN_GUI_NO_BROWSER", value)
        assert env.gui_no_browser() is True

    @pytest.mark.parametrize("value", ["", "0", "no", "false", "on", "y"])
    def test_other_values_are_false(self, monkeypatch, value):
        monkeypatch.setenv("MIMARSINAN_GUI_NO_BROWSER", value)
        assert env.gui_no_browser() is False


class TestPathDefaults:
    def test_runs_root_default(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_RUNS_ROOT", raising=False)
        assert env.runs_root() == "./generated"

    def test_runs_root_override(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "/data/runs")
        assert env.runs_root() == "/data/runs"

    def test_runs_root_empty_string_is_returned_verbatim(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "")
        assert env.runs_root() == ""

    def test_templates_dir_default(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_TEMPLATES_DIR", raising=False)
        assert env.templates_dir() == "./templates"

    def test_templates_dir_override(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", "/data/templates")
        assert env.templates_dir() == "/data/templates"


class TestFfcvCacheDir:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_FFCV_CACHE_DIR", raising=False)
        assert env.ffcv_cache_dir() is None

    def test_empty_string_is_none(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", "")
        assert env.ffcv_cache_dir() is None

    def test_set_returns_raw_value(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", "~/ffcv-cache")
        assert env.ffcv_cache_dir() == "~/ffcv-cache"


class TestImagenetRoot:
    def test_unset_is_empty_string(self, monkeypatch):
        monkeypatch.delenv("IMAGENET_ROOT", raising=False)
        assert env.imagenet_root() == ""

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv("IMAGENET_ROOT", "  /data/imagenet \n")
        assert env.imagenet_root() == "/data/imagenet"

    def test_whitespace_only_is_empty_string(self, monkeypatch):
        monkeypatch.setenv("IMAGENET_ROOT", "   ")
        assert env.imagenet_root() == ""


class TestMpStartMethod:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_MP_START_METHOD", raising=False)
        assert env.mp_start_method() is None

    def test_set_returns_value(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_MP_START_METHOD", "spawn")
        assert env.mp_start_method() == "spawn"


ALLOWED_DIRECT_READERS = {
    Path("common/env.py"),
    Path("pipelining/core/pipelines/deployment_pipeline.py"),
}


def test_no_direct_env_reads_outside_ssot():
    """Every MIMARSINAN_*/IMAGENET_ROOT environ access must go through common/env.py."""
    violations = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT)
        if rel in ALLOWED_DIRECT_READERS:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            reads_environ = "environ" in line or "getenv" in line
            if reads_environ and ("MIMARSINAN_" in line or "IMAGENET_ROOT" in line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    assert not violations, (
        "direct MIMARSINAN_* env access outside mimarsinan.common.env:\n"
        + "\n".join(violations)
    )
