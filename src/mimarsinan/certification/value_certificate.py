"""Typed value-window certificate: the value-domain (mvm) twin edge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mimarsinan.certification.count_alignment import flow_node_counts

VALUE_TWIN_FP64_ATOL = 1e-9
"""[mvm W2, measured] fp64 twin tolerance: the reassociation seams (neuron
split, pool placement, executor kernel order) measured <= 1e-12 per window on
the unit cells; 1e-9 keeps three orders of margin while sitting seven below
fp32 noise. The atol=0 cell arrives with the quantized-I/O (int-exact)
boundary contract."""


@dataclass(frozen=True)
class ValueWindowCertificate:
    """Per-neuron-window value agreement between two value-domain programs."""

    backend: str
    samples: int
    neuron_windows_compared: int
    within_atol_fraction: float
    max_abs_delta: float
    atol: float
    passed: bool

    def summary(self) -> str:
        return (
            f"backend={self.backend} samples={self.samples} "
            f"windows={self.neuron_windows_compared} "
            f"within_atol={self.within_atol_fraction:.6f} "
            f"max|delta|={self.max_abs_delta:.3e} atol={self.atol:.1e} "
            f"passed={self.passed}"
        )


def certify_twin_flow_values(
    reference_flow,
    backend_flow,
    samples: torch.Tensor,
    *,
    backend: str = "value_twin",
    atol: float = VALUE_TWIN_FP64_ATOL,
) -> tuple[ValueWindowCertificate, str]:
    """[mvm] the exact certificate edge for value programs.

    Node-granular (both programs share one IR, so node keys/widths match by
    construction — the counts-twin lesson); comparing zero windows or
    mismatched node sets never passes vacuously.
    """
    reference = flow_node_counts(reference_flow, samples)
    got = flow_node_counts(backend_flow, samples)
    only_ref = sorted(set(reference) - set(got))
    only_got = sorted(set(got) - set(reference))
    report = (
        f"nodes={len(set(reference) & set(got))} "
        f"reference-only={only_ref[:8]} backend-only={only_got[:8]}"
    )
    windows = 0
    mismatched = 0
    max_abs_delta = 0.0
    for node_id in reference:
        if node_id not in got:
            continue
        delta = (got[node_id].double() - reference[node_id].double()).abs()
        windows += int(delta.numel())
        mismatched += int((delta > atol).sum().item())
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.max().item()))
    passed = (
        not only_ref and not only_got and windows > 0 and mismatched == 0
    )
    certificate = ValueWindowCertificate(
        backend=backend,
        samples=int(samples.shape[0]),
        neuron_windows_compared=windows,
        within_atol_fraction=(
            (windows - mismatched) / windows if windows else 0.0
        ),
        max_abs_delta=max_abs_delta,
        atol=float(atol),
        passed=passed,
    )
    return certificate, report
