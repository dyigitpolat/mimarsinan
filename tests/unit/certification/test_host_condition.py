"""[W49] Theorem 1's host condition, checked by the sampled run: every host kernel the one evaluator
(``execute_compute_op_torch``) runs while the replay is armed is replayed on every sampled input alone and once more on
the whole batch, and compared at atol 0 with what it gave in the batch. A batch-dependent or nondeterministic kernel,
or one that ran on more than one device or dtype, is named and puts its deployment outside the per-sample certified
scope by name; nothing is refused, and the executor's own state (random stream, module buffers) is left as it was."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.host_condition import (
    ESTABLISHED,
    NOT_CHECKED,
    OUTSIDE_SCOPE,
    RECORD_BACKEND,
    RECORD_NAME,
    host_condition_of,
    host_kernel_replay,
)
from mimarsinan.chip_simulation.hybrid_run.host_compute import execute_compute_op_torch
from mimarsinan.deployment_record.schema import CertificateRecord
from mimarsinan.mapping.ir import ComputeOp, IRSource

WIDTH = 6


class _BatchMean(nn.Module):
    """Adds a share of the batch mean: every row moves with the samples beside it."""

    def forward(self, x):
        return x + 1e-3 * x.mean(dim=0, keepdim=True)


class _LateRows(nn.Module):
    """Moves rows 2 and later inside any batch of more than two: a replay of the first two samples alone misses it."""

    def forward(self, x):
        out = x.clone()
        if x.shape[0] > 2:
            out[2:] = out[2:] + 1e-9
        return out


class _Noisy(nn.Module):
    """A kernel that draws from the random stream: a repeated call differs."""

    def forward(self, x):
        return x + torch.rand_like(x)


def _op(module, op_id=7, name="host"):
    return ComputeOp(id=op_id, name=name,
                     input_sources=np.array([IRSource(-2, i) for i in range(WIDTH)], dtype=object),
                     op_type=type(module).__name__, params={"module": module, "input_shape": (WIDTH,)},
                     input_shape=(WIDTH,), output_shape=(WIDTH,))


def _run(op, x, dtype=torch.float64):
    with torch.no_grad():
        return execute_compute_op_torch(op, x, {-2: x}, gather_dtype=dtype)


def _batch(n=5, dtype=torch.float64, seed=0):
    return torch.randn(n, WIDTH, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_batch_invariant_kernels_establish_the_condition_on_every_sampled_input():
    x = _batch()
    with host_kernel_replay() as replay:
        _run(_op(nn.ReLU(), 1, "relu"), x)
        _run(_op(nn.Tanh(), 2, "tanh"), x)
    certificate = replay.certificate(program_host_ops=2)
    assert certificate.status == ESTABLISHED and certificate.passed
    assert certificate.samples == 5 and [k.samples for k in certificate.kernels] == [5, 5]
    assert certificate.verdict().startswith("established: ")
    assert "cpu torch.float64" in certificate.verdict()


def test_a_batch_dependent_kernel_is_named_and_outside_the_per_sample_certified_scope():
    x = _batch()
    with host_kernel_replay() as replay:
        _run(_op(nn.ReLU(), 1, "relu"), x)
        _run(_op(_BatchMean(), 2, "pool_mix"), x)
    certificate = replay.certificate(program_host_ops=2)
    assert certificate.status == OUTSIDE_SCOPE and not certificate.passed
    assert [k.name for k in certificate.batch_dependent] == ["pool_mix"]
    verdict = certificate.verdict()
    assert verdict.startswith("outside_scope: ") and "pool_mix" in verdict and "relu" not in verdict
    assert "per-sample certified scope" in verdict


def test_every_sampled_input_is_replayed_not_only_the_first_two():
    x = _batch(6)
    with host_kernel_replay() as replay:
        _run(_op(_LateRows(), 3, "late"), x)
    certificate = replay.certificate(program_host_ops=1)
    (late,) = certificate.kernels
    assert late.samples == 6 and late.batch_dependent == 4 * WIDTH
    assert certificate.status == OUTSIDE_SCOPE


def test_a_nondeterministic_kernel_is_named_and_the_random_stream_is_left_as_it_was():
    x = _batch()
    torch.manual_seed(11)
    with host_kernel_replay() as replay:
        _run(_op(_Noisy(), 4, "noisy"), x)
        after_replay = torch.rand(3)
    torch.manual_seed(11)
    _run(_op(_Noisy(), 4, "noisy"), x)
    assert torch.equal(after_replay, torch.rand(3))
    certificate = replay.certificate(program_host_ops=1)
    assert [k.name for k in certificate.nondeterministic] == ["noisy"]
    assert certificate.status == OUTSIDE_SCOPE and "nondeterministic" in certificate.verdict()


def test_a_training_mode_module_keeps_its_buffers_through_the_replay():
    norm = nn.BatchNorm1d(WIDTH).double().train()
    x = _batch()
    reference = nn.BatchNorm1d(WIDTH).double().train()
    _run(_op(reference, 5, "norm"), x)
    with host_kernel_replay() as replay:
        _run(_op(norm, 5, "norm"), x)
    assert torch.equal(norm.running_mean, reference.running_mean)
    assert int(norm.num_batches_tracked) == int(reference.num_batches_tracked) == 1
    assert replay.certificate(program_host_ops=1).status == OUTSIDE_SCOPE   # batch statistics: batch-dependent


def test_one_kernel_on_two_dtypes_is_outside_scope_by_name():
    op = _op(nn.ReLU(), 6, "relu")
    with host_kernel_replay() as replay:
        _run(op, _batch(dtype=torch.float64), dtype=torch.float64)
        _run(op, _batch(dtype=torch.float32), dtype=torch.float32)
    certificate = replay.certificate(program_host_ops=1)
    assert certificate.status == OUTSIDE_SCOPE and "one device and dtype" in certificate.verdict()
    assert [k.name for k in certificate.several_placements] == ["relu"]


def test_a_program_without_host_kernels_and_an_unexercised_one_are_told_apart():
    with host_kernel_replay() as replay:
        pass
    assert replay.certificate(program_host_ops=0).status == ESTABLISHED
    assert "no host kernel" in replay.certificate(program_host_ops=0).verdict()
    unexercised = replay.certificate(program_host_ops=3)
    assert unexercised.status == NOT_CHECKED and not unexercised.passed


def test_the_replay_is_disarmed_outside_its_block():
    x = _batch()
    with host_kernel_replay() as replay:
        pass
    _run(_op(_BatchMean(), 2, "pool_mix"), x)
    assert replay.certificate(program_host_ops=1).kernels == ()


def test_the_record_carries_the_verdict_and_the_kernels():
    x = _batch()
    with host_kernel_replay() as replay:
        _run(_op(_BatchMean(), 2, "pool_mix"), x)
    record = CertificateRecord.from_dict(replay.certificate(program_host_ops=1).record())
    assert (record.name, record.backend, record.passed) == (RECORD_NAME, RECORD_BACKEND, False)
    assert record.neuron_windows_compared == 5 * WIDTH and record.exact_match_fraction < 1.0
    detail = json.loads(record.detail)
    assert detail["verdict"].startswith("outside_scope: ")
    assert detail["kernels"][0]["name"] == "pool_mix" and detail["kernels"][0]["batch_dependent"] > 0


def test_a_nested_replay_also_feeds_the_replay_it_is_nested_in():
    """A certificate that arms its own replay (the operand certificate's host edge) inside Hard Core Mapping's leaves
    the outer replay seeing every call too."""
    x = _batch()
    with host_kernel_replay() as outer:
        with host_kernel_replay() as inner:
            _run(_op(_BatchMean(), 2, "pool_mix"), x)
        _run(_op(nn.ReLU(), 1, "relu"), x)
    assert [k.name for k in inner.certificate(program_host_ops=1).kernels] == ["pool_mix"]
    assert [k.name for k in outer.certificate(program_host_ops=2).kernels] == ["relu", "pool_mix"]
    assert outer.certificate(program_host_ops=2).status == OUTSIDE_SCOPE


def test_host_condition_of_runs_a_program_under_the_replay():
    op = _op(_BatchMean(), 2, "pool_mix")
    certificate = host_condition_of(lambda samples: _run(op, samples), _batch(), program_host_ops=1)
    assert certificate.status == OUTSIDE_SCOPE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_float_convolution_kernel_reads_by_its_measured_class(dtype):
    """A host convolution (the C5 T-LIF programs' host form): its class is whatever the replay measures on this host,
    and either way the certificate says which (no assumption that the kernel is batch-invariant)."""
    conv = nn.Conv2d(4, 8, 3, padding=1).to(dtype).eval()
    op = ComputeOp(id=9, name="conv", input_sources=np.array([IRSource(-2, i) for i in range(4 * 8 * 8)], dtype=object),
                   op_type="Conv2d", params={"module": conv, "input_shape": (4, 8, 8)},
                   input_shape=(4, 8, 8), output_shape=(8, 8, 8))
    x = torch.randn(16, 4 * 8 * 8, generator=torch.Generator().manual_seed(3)).to(dtype)
    certificate = host_condition_of(lambda samples: _run(op, samples, dtype=dtype), x, program_host_ops=1)
    (kernel,) = certificate.kernels
    assert kernel.samples == 16 and kernel.elements == 16 * 8 * 8 * 8
    assert certificate.status == (ESTABLISHED if kernel.batch_dependent == 0 else OUTSIDE_SCOPE)


class _InPlaceBatchMean(nn.Module):
    """Writes its input in place with a share of the batch mean: batch-dependent, and the gathered buffer is its output."""

    def forward(self, x):
        return x.add_(1e-3 * x.mean(dim=0, keepdim=True))


def _fresh(x):
    return x.clone()


@pytest.mark.parametrize("module", [nn.SiLU(inplace=True), nn.LeakyReLU(0.1, inplace=True), nn.Hardswish(inplace=True)],
                         ids=["silu", "leaky_relu", "hardswish"])
def test_an_in_place_host_kernel_keeps_the_deployed_output_and_is_replayed_on_its_input_as_it_came(module):
    """[review M1] A host module that writes its input in place (a form the conversion keeps): armed, the deployment seam
    returns what it returns unarmed, bit for bit, and the replay compares the kernel on the input as it came in (a
    pointwise kernel: established), not on the buffer the kernel already wrote."""
    op = _op(module, 8, "in_place")
    x = _batch()
    plain = _run(op, _fresh(x)).clone()
    with host_kernel_replay() as replay:
        armed = _run(op, _fresh(x)).clone()
    assert torch.equal(armed, plain)
    certificate = replay.certificate(program_host_ops=1)
    (kernel,) = certificate.kernels
    assert kernel.batch_dependent == 0 and kernel.nondeterministic == 0, certificate.verdict()
    assert certificate.status == ESTABLISHED and certificate.passed


def test_an_in_place_batch_dependent_kernel_is_named_and_the_deployed_output_is_unchanged():
    op = _op(_InPlaceBatchMean(), 9, "in_place_mix")
    x = _batch()
    plain = _run(op, _fresh(x)).clone()
    with host_kernel_replay() as outer:
        with host_kernel_replay() as replay:
            armed = _run(op, _fresh(x)).clone()
    assert torch.equal(armed, plain)
    for certificate in (replay.certificate(program_host_ops=1), outer.certificate(program_host_ops=1)):
        assert [k.name for k in certificate.batch_dependent] == ["in_place_mix"]
        assert certificate.status == OUTSIDE_SCOPE
        (kernel,) = certificate.kernels
        assert kernel.nondeterministic == 0          # the repeat ran on the input as it came, not on the written buffer


def test_at_one_sampled_input_batch_invariance_is_not_checked_and_the_record_does_not_pass():
    """[review M2] At a batch of one the replay alone is the batch call, so batch dependence cannot show: batch
    invariance is not checked, by name, and the record does not pass (never a vacuous established)."""
    op = _op(_BatchMean(), 2, "pool_mix")
    certificate = host_condition_of(lambda samples: _run(op, samples), _batch(1), program_host_ops=1)
    assert certificate.status == NOT_CHECKED and not certificate.passed
    (kernel,) = certificate.kernels
    assert kernel.invariance_samples == 0 and not kernel.invariance_checked
    verdict = certificate.verdict()
    assert verdict.startswith("not_checked: ") and "batch invariance" in verdict and "pool_mix" in verdict
    record = CertificateRecord.from_dict(certificate.record())
    assert not record.passed and json.loads(record.detail)["verdict"].startswith("not_checked: ")
    assert host_condition_of(lambda samples: _run(op, samples), _batch(3), program_host_ops=1).status == OUTSIDE_SCOPE


def test_at_one_sampled_input_determinism_and_one_device_and_dtype_are_still_checked():
    noisy = host_condition_of(lambda samples: _run(_op(_Noisy(), 4, "noisy"), samples), _batch(1), program_host_ops=1)
    assert noisy.status == OUTSIDE_SCOPE and [k.name for k in noisy.nondeterministic] == ["noisy"]
    op = _op(nn.ReLU(), 6, "relu")
    with host_kernel_replay() as replay:
        _run(op, _batch(1, dtype=torch.float64), dtype=torch.float64)
        _run(op, _batch(1, dtype=torch.float32), dtype=torch.float32)
    assert replay.certificate(program_host_ops=1).status == OUTSIDE_SCOPE


def test_a_kernel_seen_at_one_and_at_several_inputs_is_checked_by_its_calls_of_several():
    op = _op(nn.Tanh(), 2, "tanh")
    with host_kernel_replay() as replay:
        _run(op, _batch(1))
        _run(op, _batch(4))
    certificate = replay.certificate(program_host_ops=1)
    (kernel,) = certificate.kernels
    assert kernel.samples == 5 and kernel.invariance_samples == 4 and kernel.invariance_checked
    assert certificate.status == ESTABLISHED
