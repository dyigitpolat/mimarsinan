"""ValueWindowCertificate math: exactness, mismatch, node sets, no vacuous pass."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.value_certificate import (
    VALUE_TWIN_FP64_ATOL,
    certify_twin_flow_values,
)


@dataclass
class _Stage:
    output_map: list


@dataclass
class _Slice:
    node_id: int
    offset: int
    size: int


class _FakeFlow(nn.Module):
    """Emits fixed per-node payloads through the stage-recorder seam."""

    def __init__(self, payloads):
        super().__init__()
        self.payloads = payloads  # {node_id: (B, n) tensor}
        self.stage_count_recorder = None
        self.lif_execution_synchronized = False

    def forward(self, x):
        recorder = self.stage_count_recorder
        if recorder is not None:
            for node_id, values in self.payloads.items():
                stage = _Stage(output_map=[_Slice(node_id, 0, values.shape[1])])
                recorder(stage, values)
        return x


def _payloads(**kwargs):
    return {int(k.lstrip("n")): v for k, v in kwargs.items()}


class TestCertificate:
    def test_identical_payloads_pass(self):
        p = _payloads(n1=torch.randn(2, 3, dtype=torch.float64))
        cert, _ = certify_twin_flow_values(
            _FakeFlow(p), _FakeFlow({k: v.clone() for k, v in p.items()}),
            torch.zeros(2, 1),
        )
        assert cert.passed
        assert cert.neuron_windows_compared == 6
        assert cert.within_atol_fraction == 1.0

    def test_within_atol_perturbation_passes(self):
        base = torch.randn(2, 3, dtype=torch.float64)
        cert, _ = certify_twin_flow_values(
            _FakeFlow(_payloads(n1=base)),
            _FakeFlow(_payloads(n1=base + VALUE_TWIN_FP64_ATOL / 10)),
            torch.zeros(2, 1),
        )
        assert cert.passed

    def test_mismatch_fails_with_measured_delta(self):
        # Seeded base and a dyadic step: b + 2**-10 - b is exactly 2**-10 here, whatever ran before.
        base = torch.randn(2, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        step = 2.0 ** -10
        got = base.clone()
        got[0, 0] += step
        assert float(got[0, 0] - base[0, 0]) == step
        cert, _ = certify_twin_flow_values(
            _FakeFlow(_payloads(n1=base)), _FakeFlow(_payloads(n1=got)),
            torch.zeros(2, 1),
        )
        assert not cert.passed
        assert cert.max_abs_delta == step
        assert cert.within_atol_fraction < 1.0

    def test_node_set_mismatch_fails(self):
        base = torch.randn(2, 3, dtype=torch.float64)
        cert, report = certify_twin_flow_values(
            _FakeFlow(_payloads(n1=base, n2=base)),
            _FakeFlow(_payloads(n1=base)),
            torch.zeros(2, 1),
        )
        assert not cert.passed
        assert "reference-only=[2]" in report

    def test_zero_windows_never_pass_vacuously(self):
        cert, _ = certify_twin_flow_values(
            _FakeFlow({}), _FakeFlow({}), torch.zeros(2, 1)
        )
        assert not cert.passed
        assert cert.neuron_windows_compared == 0
