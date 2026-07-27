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

VALUE_R_EDGE_WQ_ATOL = 1e-5
"""[mvm W3, measured in vivo t0_41] the honest R-edge residual under weight
quantization: the model carries fp32-PROJECTED weights (round(w*s)/s in
fp32) while the chip program computes integer-grid / scale in fp64, so
model↔identity differs by the fp32 representation of the ratio (measured
max|delta| 1.0e-7 on lenet5 logits; 100x margin). The C-edge stays at
VALUE_TWIN_FP64_ATOL — both twins share the integer convention. This is the
value-domain instance of the §17 rule: (R) model↔grid is the honest WQ
residual, (C) twin↔twin is the exact edge."""


VALUE_R_EDGE_AQ_LSB_BOUND = 1.0
"""[mvm AQ, measured on t1_11] Under boundary activation quantization the
model↔chip relation is PIECEWISE CONSTANT, so a scalar atol is the wrong
instrument: the unavoidable fp seed (the model holds fp32 weights, the chip
accumulates integers then divides by theta — 2.1e-08 relative at hop 1)
crosses grid edges and amplifies to ~1 LSB over a deep chain (measured
9.8e-03 out of a 2.7e-02 LSB across 26 hops, every node affected). The
honest R-edge for an AQ program is therefore judged in GRID UNITS — the
divergence must stay within one boundary LSB — together with exact decision
parity, which is what a deployed classifier actually promises. Both remain
FATAL; only the unit changed."""


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
