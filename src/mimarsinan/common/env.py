"""Single source of truth for MIMARSINAN_* (and IMAGENET_ROOT) environment variables.

Every accessor reads ``os.environ`` at call time so tests can monkeypatch it.
"""

from __future__ import annotations

import os

CUDA_DEBUG_VAR = "MIMARSINAN_CUDA_DEBUG"
VRAM_PROBE_VAR = "MIMARSINAN_VRAM_PROBE"
RESOURCE_DEBUG_VAR = "MIMARSINAN_RESOURCE_DEBUG"
NF_SCM_PARITY_DEBUG_VAR = "MIMARSINAN_NF_SCM_PARITY_DEBUG"
DISABLE_FFCV_VAR = "MIMARSINAN_DISABLE_FFCV"
FFCV_CACHE_DIR_VAR = "MIMARSINAN_FFCV_CACHE_DIR"
LOIHI_QUIET_VAR = "MIMARSINAN_LOIHI_QUIET"
GUI_NO_BROWSER_VAR = "MIMARSINAN_GUI_NO_BROWSER"
RUNS_ROOT_VAR = "MIMARSINAN_RUNS_ROOT"
TEMPLATES_DIR_VAR = "MIMARSINAN_TEMPLATES_DIR"
TEST_CUDA_VAR = "MIMARSINAN_TEST_CUDA"
MP_START_METHOD_VAR = "MIMARSINAN_MP_START_METHOD"
MBH_LEDGER_VAR = "MIMARSINAN_MBH_LEDGER"
MBH_GATE_VAR = "MIMARSINAN_MBH_GATE"
MBH_LIF_REALLOC_VAR = "MIMARSINAN_MBH_LIF_REALLOC"
MBH_LIF_TANNEAL_VAR = "MIMARSINAN_MBH_LIF_TANNEAL"
IMAGENET_ROOT_VAR = "IMAGENET_ROOT"


def cuda_debug_enabled() -> bool:
    """Synchronous CUDA debug checks are on (value exactly "1")."""
    return os.environ.get(CUDA_DEBUG_VAR) == "1"


def set_cuda_debug(enabled: bool = True) -> None:
    """Set or clear the CUDA debug flag for this process and its children."""
    if enabled:
        os.environ[CUDA_DEBUG_VAR] = "1"
    else:
        os.environ.pop(CUDA_DEBUG_VAR, None)


def vram_probe_enabled() -> bool:
    """Opt-in VRAM/RSS probes are on (value exactly "1")."""
    return os.environ.get(VRAM_PROBE_VAR) == "1"


def resource_debug_enabled() -> bool:
    """Per-step process resource snapshots are on (value exactly "1")."""
    return os.environ.get(RESOURCE_DEBUG_VAR) == "1"


def nf_scm_parity_debug_enabled() -> bool:
    """Verbose NF-SCM parity-gate diagnostics are on (value exactly "1")."""
    return os.environ.get(NF_SCM_PARITY_DEBUG_VAR) == "1"


def ffcv_disabled() -> bool:
    """FFCV data loading is force-disabled (value exactly "1")."""
    return os.environ.get(DISABLE_FFCV_VAR) == "1"


def ffcv_cache_dir() -> str | None:
    """Override for the FFCV beton cache root; None when unset or empty."""
    return os.environ.get(FFCV_CACHE_DIR_VAR) or None


def loihi_quiet() -> bool:
    """Per-core Lava-Loihi runner logging is silenced (value exactly "1")."""
    return os.environ.get(LOIHI_QUIET_VAR) == "1"


def gui_no_browser() -> bool:
    """GUI server skips opening a browser ("1"/"true"/"yes", case/space-insensitive)."""
    return os.environ.get(GUI_NO_BROWSER_VAR, "").strip().lower() in ("1", "true", "yes")


def runs_root() -> str:
    """Directory holding past pipeline runs; defaults to "./generated"."""
    return os.environ.get(RUNS_ROOT_VAR, "./generated")


def templates_dir() -> str:
    """Directory holding saved config templates; defaults to "./templates"."""
    return os.environ.get(TEMPLATES_DIR_VAR, "./templates")


def test_cuda_enabled() -> bool:
    """Test suite may use CUDA (value exactly "1"); read by tests/conftest.py."""
    return os.environ.get(TEST_CUDA_VAR) == "1"


def mbh_ledger_enabled() -> bool:
    """The measurement-only per-rung [MBH] fast-ladder ledger is on (value exactly "1")."""
    return os.environ.get(MBH_LEDGER_VAR) == "1"


def mbh_gate_enabled() -> bool:
    """The [MBH-GATE] D-hat-gated fast ladder (X2/E1) is on (value exactly "1")."""
    return os.environ.get(MBH_GATE_VAR) == "1"


def mbh_lif_realloc_enabled() -> bool:
    """The MBH LIF wall-reallocation experiment (X2/E2) is on (value exactly "1")."""
    return os.environ.get(MBH_LIF_REALLOC_VAR) == "1"


def mbh_lif_tanneal_enabled() -> bool:
    """The MBH LIF T-annealing realizable family (X2b, T1/P1') is on (value exactly "1")."""
    return os.environ.get(MBH_LIF_TANNEAL_VAR) == "1"


def mp_start_method() -> str | None:
    """Multiprocessing start-method override read by the src/init.py bootstrap."""
    return os.environ.get(MP_START_METHOD_VAR)


def imagenet_root() -> str:
    """ILSVRC2012 root directory, stripped; empty string when unset."""
    return os.environ.get(IMAGENET_ROOT_VAR, "").strip()
