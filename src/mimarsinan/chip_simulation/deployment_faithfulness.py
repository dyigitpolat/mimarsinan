"""Pure-data registry of standing deployment-faithfulness gates and external-dependency boundary guards."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from mimarsinan.chip_simulation.spiking_semantics import (
    _BACKEND_CAPS,
    backend_capabilities,
    require_spiking_mode_supported,
)


@dataclass(frozen=True)
class FaithfulnessGate:
    """A gate that runs on a deployment run to keep the deployed number honest.

    ``default_on`` marks a *standing* gate (locked default-ON by the audit test, so it
    cannot regress to opt-in by a silent defaults edit). ``fatal`` marks the rows
    that can FAIL a run; a standing row that is not fatal is a standing REPORT.
    """

    name: str
    config_flag: str
    default_on: bool
    description: str
    fatal: bool = True


DEPLOYMENT_FAITHFULNESS_GATES: Tuple[FaithfulnessGate, ...] = (
    FaithfulnessGate(
        name="readout_decision_drift",
        config_flag="scm_torch_sim_parity_check",
        default_on=True,
        # Fatal exactly where it is the hop's only guard: sync families, where
        # the NF-SCM exactness gate does not arm. Under streamed lif it does
        # not run at all (exactness holds the same hop at atol=0).
        fatal=True,
        description=(
            "How much of the readout the trained torch NF and the EXACT spiking sim "
            "run_scm_identity_metric deploys (build_spiking_hybrid_flow) place on the "
            "same integer count, both read inside the chip-lattice measurement plane. "
            "A floor-gated observable: report-only where exactness arms, fatal where "
            "it is the sole guard. The two are different float programs, and "
            "reading them as an argmax agreement measured float32 tie-breaking on a "
            "tie-dense integer readout instead (narrowconv 0.9883 = 3 flips, all on "
            "IDENTICAL counts). It does not arm for streamed lif, where the count "
            "exactness gate holds the same hop at atol=0. Wired in "
            "SoftCoreMappingStep._run_readout_decision_drift_diagnostic."
        ),
    ),
    FaithfulnessGate(
        name="nf_scm_per_neuron_parity",
        config_flag="nf_scm_parity_samples",
        default_on=True,
        description=(
            "Per-neuron NF<->SCM lock for analytic schedules (and a decision-level "
            "argmax gate for cascaded). Accuracy tolerance alone hid a 3.8 pp "
            "semantic split (2026-06-06); this fails loud naming the diverging "
            "neuron. Wired in SoftCoreMappingStep._run_nf_scm_parity_gate."
        ),
    ),
)


def standing_gates() -> Tuple[FaithfulnessGate, ...]:
    """The gates that must run on every deployment run (default-on)."""
    return tuple(g for g in DEPLOYMENT_FAITHFULNESS_GATES if g.default_on)


GUARD_KINDS = ("version_pin", "capability_gate", "lazy_import")


@dataclass(frozen=True)
class DependencyBoundary:
    """An external-dependency import site that can break a deployment silently.

    ``guards`` is the non-empty set of guard kinds protecting it; ``verify`` (optional)
    asserts the guard is actually present. An empty ``guards`` set fails the checklist loud.
    """

    package: str
    integration_module: str
    guards: Tuple[str, ...]
    rationale: str
    verify: Optional[Callable[[], None]] = None


def _verify_sanafe_guard() -> None:
    from mimarsinan.chip_simulation.sanafe.arch_synth.spec import (
        _check_sanafe_version,
        _SUPPORTED_SANAFE_VERSIONS,
    )

    if not _SUPPORTED_SANAFE_VERSIONS:
        raise AssertionError("SANA-FE supported-version pin is empty")
    _check_sanafe_version(_SUPPORTED_SANAFE_VERSIONS[0])
    bad = _bump_patch(_SUPPORTED_SANAFE_VERSIONS[-1])
    try:
        _check_sanafe_version(bad)
    except RuntimeError:
        return
    raise AssertionError(
        f"SANA-FE version guard did not reject the unsupported version {bad!r}"
    )


def _verify_backend_capability_guard(backend: str) -> Callable[[], None]:
    def _verify() -> None:
        if backend not in _BACKEND_CAPS:
            raise AssertionError(
                f"backend {backend!r} has no declared capabilities in "
                f"_BACKEND_CAPS — the capability gate would fall through to the "
                f"permissive default and a wrong (firing x mode) would not fail loud"
            )
        caps = backend_capabilities(backend)
        assert caps is not None
        assert callable(require_spiking_mode_supported)

    return _verify


EXTERNAL_DEPENDENCY_BOUNDARIES: Tuple[DependencyBoundary, ...] = (
    DependencyBoundary(
        package="sanafe",
        integration_module="mimarsinan.chip_simulation.sanafe.arch_synth.spec",
        guards=("version_pin", "capability_gate", "lazy_import"),
        rationale=(
            "GPL-3.0 detailed-stats backend. An unpinned upgrade to 2.2.x SIGFPEs "
            "in C++ on arch load (2026-06-17). _check_sanafe_version fails loud on "
            "an unsupported version; the capability registry gates (firing x mode); "
            "the import is lazy so importing mimarsinan never requires it."
        ),
        verify=_verify_sanafe_guard,
    ),
    DependencyBoundary(
        package="lava",
        integration_module="mimarsinan.chip_simulation.lava_loihi.core_lava",
        guards=("capability_gate", "lazy_import"),
        rationale=(
            "Optional Loihi simulator. lava is LIF-only — the capability registry "
            "blocks every TTFS mode on the lava/loihi backends (behavior."
            "require_backend('lava')); the import is lazy inside _run_core_lava."
        ),
        verify=_verify_backend_capability_guard("lava"),
    ),
    DependencyBoundary(
        package="pyxrt",
        integration_module="mimarsinan.chip_simulation.odin_fpga.xrt_transport",
        guards=("capability_gate", "lazy_import"),
        rationale=(
            "The Xilinx runtime's Python binding, which reaches a physical "
            "Alveo board. It ships with XRT and never with pip, so the import "
            "lives inside load_pyxrt() and every entry refuses by name when it "
            "is absent — importing mimarsinan on a machine with no Alveo "
            "installation must keep working. The capability registry gates the "
            "backend itself: odin_fpga declares which (firing x mode x soma "
            "law) the crossbar executes, so a board can never be handed a law "
            "it does not implement. There is no version pin: XRT's version is "
            "the CLUSTER's (Vitis 2022.2 on HACC), the shell is pinned by the "
            "xclbin the run loads, and pinning a host library we do not "
            "install would be a claim we cannot enforce."
        ),
        verify=_verify_backend_capability_guard("odin_fpga"),
    ),
    DependencyBoundary(
        package="ffcv",
        integration_module="mimarsinan.data_handling.ffcv.loader_factory",
        guards=("lazy_import",),
        rationale=(
            "Optional fast data loader. Imported lazily inside the ffcv submodule; "
            "a missing install surfaces a normal ImportError and the torch "
            "DataLoader path is used. It feeds data only, never the deployed "
            "forward, so it cannot move a deployment number."
        ),
    ),
)


def boundary_for(package: str) -> Optional[DependencyBoundary]:
    for b in EXTERNAL_DEPENDENCY_BOUNDARIES:
        if b.package == package:
            return b
    return None


def sanafe_supported_versions() -> Tuple[str, ...]:
    """The SSOT pinned-SANA-FE version tuple (re-exported from the guard)."""
    from mimarsinan.chip_simulation.sanafe.arch_synth.spec import (
        _SUPPORTED_SANAFE_VERSIONS,
    )

    return tuple(_SUPPORTED_SANAFE_VERSIONS)


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", ".."))


def bootstrap_pinned_sanafe_version(script_path: Optional[str] = None) -> Optional[str]:
    """The ``sanafe==X.Y.Z`` version the bootstrap script installs (or ``None``)."""
    if script_path is None:
        script_path = os.path.join(_project_root(), "scripts", "bootstrap_sanafe.sh")
    if not os.path.isfile(script_path):
        return None
    with open(script_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"sanafe==([0-9][0-9A-Za-z.\-]*)", text)
    return match.group(1) if match else None


def assert_sanafe_pin_consistent(script_path: Optional[str] = None) -> str:
    """Fail loud if the bootstrap-script pin drifts from the code guard's SSOT; return the agreed version."""
    supported = sanafe_supported_versions()
    if not supported:
        raise AssertionError("SANA-FE supported-version pin is empty")
    script_pin = bootstrap_pinned_sanafe_version(script_path)
    if script_pin is None:
        raise AssertionError(
            "scripts/bootstrap_sanafe.sh declares no `sanafe==<version>` pin — "
            "the bootstrap could install an unguarded version"
        )
    if script_pin not in supported:
        raise AssertionError(
            f"SANA-FE pin drift: bootstrap installs {script_pin!r} but the code "
            f"guard _SUPPORTED_SANAFE_VERSIONS={supported!r}. Re-validate the "
            f"SANA-FE parity gate and bump BOTH together."
        )
    return script_pin


def _bump_patch(version: str) -> str:
    """A version one patch beyond ``version`` (for a known-unsupported probe)."""
    parts = version.split(".")
    if parts and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    return version + ".999"


DEPLOYED_METRIC_PROTOCOL = {
    "metric_entrypoint": "run_scm_identity_metric",
    "deployed_executor_builder": "build_spiking_hybrid_flow",
    "parity_gate": "assert_streamed_nf_scm_exact_or_raise",
    "readout_drift_report": "measure_readout_decision_drift",
    "metric_step": "SoftCoreMappingStep",
}


__all__ = [
    "FaithfulnessGate",
    "DEPLOYMENT_FAITHFULNESS_GATES",
    "standing_gates",
    "GUARD_KINDS",
    "DependencyBoundary",
    "EXTERNAL_DEPENDENCY_BOUNDARIES",
    "boundary_for",
    "sanafe_supported_versions",
    "bootstrap_pinned_sanafe_version",
    "assert_sanafe_pin_consistent",
    "DEPLOYED_METRIC_PROTOCOL",
]
