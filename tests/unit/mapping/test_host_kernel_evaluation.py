"""[U-PS P-1] The declared host-kernel evaluation of a deployed program.

``host_kernel_evaluation = per_sample`` makes the program evaluate every host kernel one sample at a time: a loop over
the batch at the seam every executor calls, each sample on its own copy, the results concatenated in order. A
batch-dependent fp32 host convolution or matmul then reads Theorem 1's host condition established by construction,
where on its batch it reads outside scope; its outputs equal the per-sample replay's at atol 0. Absent (or
``batched``), every program and every output is the base's, byte for byte. A kernel that cannot run on one sample
alone is refused by name. The fp32 kernels below are pinned by their measured class on this host (U-X X-2): a host on
which they are batch-invariant skips by name, since there is nothing to remedy there."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.host_condition import ESTABLISHED, OUTSIDE_SCOPE, host_kernel_replay
from mimarsinan.chip_simulation.hybrid_run.host_compute import execute_compute_op_numpy, execute_compute_op_torch
from mimarsinan.chip_simulation.value_run.host_ops import execute_host_op
from mimarsinan.common.refusal import is_toolchain_refusal
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource
from mimarsinan.mapping.support.host_kernel_evaluation import (
    BATCHED,
    HOST_KERNEL_EVALUATION_KEY,
    PER_SAMPLE,
    PerSampleHostKernelRefusal,
    declare_host_kernel_evaluation,
    declared_evaluation,
    forward_of,
    program_evaluates_per_sample,
    twin_as_program,
)
from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC

BATCH = 8


def _conv():
    torch.manual_seed(0)
    return nn.Conv2d(3, 16, 3, padding=1).eval()


def _linear():
    torch.manual_seed(0)
    return nn.Linear(120, 84).eval()


#: The C5 T-LIF programs' host forms (U-TT T-6): an fp32 convolution and an fp32 classifier matmul (``mms``).
KERNELS = {"conv2d": (_conv, (3, 32, 32)), "linear": (_linear, (120,))}


def _width(shape):
    return int(np.prod(shape))


def _op(module, shape, op_id=3, name="host_kernel"):
    width = _width(shape)
    return ComputeOp(id=op_id, name=name,
                     input_sources=np.array([IRSource(-2, i) for i in range(width)], dtype=object),
                     op_type=type(module).__name__, params={"module": module, "input_shape": tuple(shape)},
                     input_shape=tuple(shape))


def _program(op):
    return IRGraph(nodes=[op], output_sources=np.array([IRSource(op.id, 0)], dtype=object))


def _batch(shape, n=BATCH, seed=1):
    return torch.randn(n, _width(shape), generator=torch.Generator().manual_seed(seed))


def _run(op, x):
    with torch.no_grad():
        return execute_compute_op_torch(op, x, {-2: x}, arithmetic=SPIKING_ARITHMETIC)


def _alone(module, shape, x):
    """Every sample alone through the bare module, concatenated: the per-sample replay's reading."""
    with torch.no_grad():
        return torch.cat([module(x[i:i + 1].clone().view(1, *shape)).reshape(1, -1) for i in range(x.shape[0])])


def _batch_dependent_here(module, shape, x):
    with torch.no_grad():
        batched = module(x.view(x.shape[0], *shape)).reshape(x.shape[0], -1)
    return bool((batched != _alone(module, shape, x)).any())


@pytest.fixture(params=sorted(KERNELS))
def kernel(request):
    make, shape = KERNELS[request.param]
    module, x = make(), _batch(shape)
    if not _batch_dependent_here(module, shape, x):
        pytest.skip(f"fp32 {request.param} is batch-invariant on this host at batch {BATCH}: nothing to remedy")
    return request.param, module, shape, x


class TestTheHostConditionByConstruction:
    def test_a_batch_dependent_fp32_kernel_reads_outside_scope_with_the_key_off(self, kernel):
        name, module, shape, x = kernel
        op = _op(module, shape, name=name)
        with host_kernel_replay() as replay:
            _run(op, x)
        certificate = replay.certificate(program_host_ops=1)
        assert certificate.status == OUTSIDE_SCOPE and not certificate.passed
        assert [k.name for k in certificate.batch_dependent] == [name]
        assert certificate.per_sample_ops == ()

    def test_the_same_kernel_reads_established_with_the_key_on(self, kernel):
        name, module, shape, x = kernel
        op = _op(module, shape, name=name)
        assert declare_host_kernel_evaluation(_program(op), PER_SAMPLE) == 1
        with host_kernel_replay() as replay:
            _run(op, x)
        certificate = replay.certificate(program_host_ops=1)
        assert certificate.status == ESTABLISHED and certificate.passed
        (reading,) = certificate.kernels
        assert reading.batch_dependent == 0 and reading.nondeterministic == 0 and reading.max_abs_delta == 0.0
        assert reading.invariance_samples == BATCH, "batch invariance is checked on the batch, not assumed"
        assert certificate.per_sample_ops == (op.id,)
        assert f"{HOST_KERNEL_EVALUATION_KEY}={PER_SAMPLE}" in certificate.verdict()
        record = certificate.record()
        assert record["passed"] is True
        assert json.loads(record["detail"])["per_sample_ops"] == [op.id]

    def test_outputs_with_the_key_on_equal_the_per_sample_replay_at_atol_0(self, kernel):
        name, module, shape, x = kernel
        op = _op(module, shape, name=name)
        batched = _run(op, x)
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        got = _run(op, x)
        assert torch.equal(got, _alone(module, shape, x))
        assert not torch.equal(got, batched), "the remedy must change what a batch-dependent kernel gives"


class TestTheKeyOffIsByteIdentical:
    @pytest.mark.parametrize("mode", [None, BATCHED])
    def test_absent_or_batched_stamps_nothing_and_the_program_pickles_as_before(self, mode):
        module, shape = _linear(), (120,)
        op = _op(module, shape)
        program = _program(op)
        before = pickle.dumps(program)
        params = dict(op.params)
        assert declare_host_kernel_evaluation(program, mode) == 0
        assert op.params == params and HOST_KERNEL_EVALUATION_KEY not in op.params
        assert pickle.dumps(program) == before
        assert declared_evaluation(op) == BATCHED and not program_evaluates_per_sample(program)

    @pytest.mark.parametrize("name", sorted(KERNELS))
    def test_outputs_with_the_key_off_are_the_batch_call_bit_for_bit(self, name):
        make, shape = KERNELS[name]
        module, x = make(), _batch(shape)
        op = _op(module, shape)
        with torch.no_grad():
            want = module(x.view(x.shape[0], *shape)).reshape(x.shape[0], -1)
        assert torch.equal(_run(op, x), want)
        assert torch.equal(op.execute_on_gathered(x), want)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match=HOST_KERNEL_EVALUATION_KEY):
            declare_host_kernel_evaluation(_program(_op(_linear(), (120,))), "per_batch")


class TestEveryExecutorAlike:
    def test_every_evaluation_seam_follows_the_declaration(self):
        module, shape = _linear(), (120,)
        x = _batch(shape)
        op = _op(module, shape)
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        want = _alone(module, shape, x)
        with torch.no_grad():
            seams = {
                "ComputeOp.execute": op.execute(x, {}),
                "ComputeOp.execute_on_gathered": op.execute_on_gathered(x),
                "ComputeOp.probe_on_gathered": op.probe_on_gathered(x),
                "execute_compute_op_torch": _run(op, x),
                "execute_compute_op_numpy": torch.from_numpy(
                    execute_compute_op_numpy(op, x.numpy(), {-2: x.numpy()}, device="cpu", dtype=np.float32)),
                "value_run.execute_host_op": execute_host_op(op, x, {-2: x}, dtype=torch.float32, twins={}),
            }
        assert {k: bool(torch.equal(v, want)) for k, v in seams.items()} == {k: True for k in seams}

    def test_the_kernel_sees_one_sample_per_call(self):
        module, shape = _conv(), (3, 32, 32)
        seen = []
        module.register_forward_pre_hook(lambda _m, args: seen.append(int(args[0].shape[0])))
        op = _op(module, shape)
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        _run(op, _batch(shape))
        assert seen == [1] * BATCH

    def test_the_declaration_rides_on_the_program_through_its_pickle(self):
        module, shape = _linear(), (120,)
        x = _batch(shape)
        program = _program(_op(module, shape))
        declare_host_kernel_evaluation(program, PER_SAMPLE)
        loaded = pickle.loads(pickle.dumps(program))
        assert program_evaluates_per_sample(loaded)
        assert torch.equal(_run(loaded.get_compute_ops()[0], x), _alone(module, shape, x))

    def test_an_empty_batch_runs_as_given(self):
        module, shape = _linear(), (120,)
        op = _op(module, shape)
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        assert tuple(_run(op, _batch(shape, n=0)).shape) == (0, 84)


class _BatchFolding(nn.Module):
    """Gives two rows for each one it is handed: no per-sample function."""

    def forward(self, x):
        return x.repeat(2, 1)


class TestARefusalByName:
    def test_a_kernel_that_cannot_run_on_one_sample_is_refused_by_name(self):
        shape = (120,)
        norm = nn.BatchNorm1d(120).train()  # batch statistics: undefined on one sample
        op = _op(norm, shape, name="batch_stat_norm")
        x = _batch(shape)
        _run(op, x)  # on its batch it runs
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        with pytest.raises(PerSampleHostKernelRefusal, match="batch_stat_norm") as caught:
            _run(op, x)
        assert is_toolchain_refusal(caught.value)
        assert "cannot be evaluated on one sample alone" in str(caught.value)

    def test_a_kernel_that_folds_the_batch_is_refused_by_name(self):
        shape = (6,)
        op = _op(_BatchFolding(), shape, name="batch_folding")
        declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
        with pytest.raises(PerSampleHostKernelRefusal, match="batch_folding.*for one sample"):
            _run(op, _batch(shape))


class TestTheTwinFollowsTheProgram:
    def test_the_twin_is_evaluated_one_sample_at_a_time_where_the_program_declares_it(self):
        module, shape = _linear(), (120,)
        x = _batch(shape)
        program = _program(_op(module, shape))
        with torch.no_grad():
            assert torch.equal(twin_as_program(program, forward_of, module, x), module(x))
            declare_host_kernel_evaluation(program, PER_SAMPLE)
            assert torch.equal(twin_as_program(program, forward_of, module, x), _alone(module, shape, x))

    def test_per_key_captures_concatenate_in_order(self):
        program = _program(_op(_linear(), (120,)))
        declare_host_kernel_evaluation(program, PER_SAMPLE)
        x = _batch((120,), n=5)
        got = twin_as_program(program, lambda _m, s: {0: s.numpy()[:, :3], 1: s[:, 3:4]}, None, x)
        assert np.array_equal(got[0], x.numpy()[:, :3]) and torch.equal(got[1], x[:, 3:4])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA device visible: the GPU form runs only on a qualifying sura GPU (U-PS real_cuda_run)")
@pytest.mark.parametrize("name", sorted(KERNELS))
def test_on_a_gpu_the_declared_kernel_reads_established_and_equals_the_per_sample_replay(name):
    """The tuned rows' sessions evaluate their sampled certificates on a GPU: there too the declared kernel is a
    function of each sample alone (and the repeat equal), whatever the batch reading of the bare kernel is."""
    make, shape = KERNELS[name]
    module, x = make().cuda(), _batch(shape).cuda()
    op = _op(module, shape, name=name)
    with host_kernel_replay() as replay:
        _run(op, x)
    off = replay.certificate(program_host_ops=1)
    declare_host_kernel_evaluation(_program(op), PER_SAMPLE)
    with host_kernel_replay() as replay:
        got = _run(op, x)
    on = replay.certificate(program_host_ops=1)
    assert on.status == ESTABLISHED and on.passed, (on.verdict(), off.verdict())
    assert torch.equal(got, _alone(module, shape, x))
    print(f"[U-PS] cuda {name}: key off {off.status} ({off.kernels[0].batch_dependent} of {off.kernels[0].elements} "
          f"elements batch-dependent); key on {on.status}")
