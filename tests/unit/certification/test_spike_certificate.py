"""[calculus §17/PR42] the spike-count certificate: per-neuron window counts,
exact by construction for integer-contract backends; typed, fail-loud."""

from __future__ import annotations

import torch

from mimarsinan.certification.spike_certificate import (
    SpikeCountCertificate,
    certify_spike_counts,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW


def _ref(_batch):
    return {("seg0", 0): torch.tensor([[3.0, 0.0, 7.0]]),
            ("seg1", 0): torch.tensor([[1.0, 2.0]])}


def test_exact_backend_passes_on_identity():
    cert = certify_spike_counts(_ref, _ref, [torch.zeros(1, 4)],
                                backend="hcm")
    assert isinstance(cert, SpikeCountCertificate)
    assert cert.passed and cert.exact_match_fraction == 1.0
    assert cert.backend_class == "exact"
    assert cert.neuron_windows_compared == 5
    assert cert.max_abs_delta == 0.0


def test_exact_backend_fails_loud_on_one_count_flip():
    def backend(_batch):
        d = _ref(_batch)
        d[("seg0", 0)] = d[("seg0", 0)].clone()
        d[("seg0", 0)][0, 2] += 1.0
        return d

    cert = certify_spike_counts(_ref, backend, [torch.zeros(1, 4)],
                                backend="hcm")
    assert not cert.passed
    assert cert.max_abs_delta == 1.0
    assert cert.exact_match_fraction < 1.0
    assert cert.divergent and cert.divergent[0][0] == ("seg0", 0)


def test_counts_export_class_uses_tolerance():
    def backend(_batch):
        d = _ref(_batch)
        d[("seg1", 0)] = d[("seg1", 0)] + 1.0
        return d

    strict = certify_spike_counts(_ref, backend, [torch.zeros(1, 4)],
                                  backend="loihi")
    assert strict.backend_class == "counts-export"
    assert strict.passed  # documented tolerance 1 count for counts-export
    exact = certify_spike_counts(_ref, backend, [torch.zeros(1, 4)],
                                 backend="hcm")
    assert not exact.passed


def test_unknown_backend_fails_loud():
    import pytest

    with pytest.raises(KeyError, match="backend"):
        certify_spike_counts(_ref, _ref, [torch.zeros(1, 4)],
                             backend="mystery-chip")


def test_integration_nf_walk_vs_hcm_counts_on_the_tiny_fixture():
    import sys

    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import _tiny, _flow, T as _T
    import mimarsinan.models.spiking.hybrid.lif_step as ls
    from mimarsinan.models.spiking.hybrid.executors.sync_counts import (
        run_neural_segment_counts,
    )
    from mimarsinan.spiking.segment_forward import (
        LifSegmentPolicy, SegmentForwardDriver,
    )

    repr_, hybrid = _tiny()
    percs = list(repr_.get_perceptrons())

    def reference(batch):
        driver = SegmentForwardDriver(
            repr_, _T, LifSegmentPolicy(synchronized=True, soma_law=DEFAULT_SOMA_LAW))
        rec = {}
        with torch.no_grad():
            driver(batch, node_value_recorder=rec)
        out = {}
        for k, p in enumerate(percs):
            theta = float(torch.as_tensor(p.activation_scale).float().mean())
            out[("hop", k)] = rec[id(p)] / theta * _T
        return {("hop", len(percs) - 1): out[("hop", len(percs) - 1)]}

    def backend(batch):
        calls = []
        orig = run_neural_segment_counts

        def spy(f, train, **kw):
            r = {}
            o = orig(f, train, neuron_count_recorder=r, **kw)
            calls.append(r)
            return o

        ls.run_neural_segment_counts = spy
        try:
            with torch.no_grad():
                _flow(hybrid, synchronized=True)(batch)
        finally:
            ls.run_neural_segment_counts = orig
        last = calls[-1]
        return {("hop", len(percs) - 1): torch.cat(
            [last[i] for i in sorted(last)], dim=1)}

    x = torch.rand(2, 8) * 0.9
    cert = certify_spike_counts(reference, backend, [x], backend="hcm")
    assert cert.passed, cert.summary()
    assert cert.exact_match_fraction == 1.0


def test_zero_comparisons_is_not_a_certificate():
    import pytest

    with pytest.raises(ValueError, match="ZERO neuron-windows"):
        certify_spike_counts(lambda b: {}, lambda b: {}, [torch.zeros(1, 4)],
                             backend="hcm")
