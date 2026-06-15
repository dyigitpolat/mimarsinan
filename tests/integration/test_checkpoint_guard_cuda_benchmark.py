"""IV.7 (large-model half): CheckpointGuard scope/location budgets on CUDA.

The deterministic IV.7 gates (``tests/unit/tuning/transformation/test_perf_gates.py``)
cover probe-count and the tunable-scope key skip CPU-side. This is the CUDA half
the code review flagged as not-yet-a-gate: on a large frozen-backbone model (the
partial-fine-tune shape a ViT clamp probe exhibits), ``scope="tunable"`` must
free the frozen backbone's bytes and ``location="cpu_pinned"`` must free device
VRAM via a real, faster-than-pageable async offload (W6).

Marked ``slow`` (allocates ~0.8 GB on the GPU) and skipped without CUDA.
"""

import time

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.slow


class _BackboneHead(nn.Module):
    """Large frozen backbone + small tunable head — the W6 partial-FT shape."""

    def __init__(self, width=4096, depth=12, head=256):
        super().__init__()
        self.backbone = nn.Sequential(*[nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, head)
        for p in self.backbone.parameters():
            p.requires_grad_(False)


class _Stub:
    def __init__(self, model):
        self.model = model
        self.device = "cuda"


def _bytes(handle):
    return sum(v.numel() * v.element_size() for v in handle.state.values())


def _cuda_model():
    return _BackboneHead().to("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tunable_scope_frees_frozen_backbone_bytes():
    from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard

    stub = _Stub(_cuda_model())
    full = CheckpointGuard(stub, scope="full", location="device").snapshot()
    tunable = CheckpointGuard(stub, scope="tunable", location="device").snapshot()
    # The frozen backbone dominates → tunable captures only the tiny head.
    assert _bytes(tunable) < 0.05 * _bytes(full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_pinned_frees_device_vram():
    from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard

    stub = _Stub(_cuda_model())
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    base = torch.cuda.memory_allocated()
    dev = CheckpointGuard(stub, scope="full", location="device").snapshot()
    torch.cuda.synchronize()
    dev_extra = torch.cuda.memory_allocated() - base
    del dev
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    base = torch.cuda.memory_allocated()
    pin = CheckpointGuard(stub, scope="full", location="cpu_pinned").snapshot()
    torch.cuda.synchronize()
    pin_extra = torch.cuda.memory_allocated() - base

    assert dev_extra > 0  # a device snapshot roughly doubles model VRAM
    assert pin_extra <= 0.05 * dev_extra  # pinned offloads it to host RAM
    del pin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_pinned_async_is_faster_than_pageable():
    from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard

    model = _cuda_model()
    stub = _Stub(model)
    guard = CheckpointGuard(stub, scope="full", location="cpu_pinned")

    def pageable():
        st = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        _ = {k: v.to("cuda") for k, v in st.items()}
        torch.cuda.synchronize()

    def pinned():
        handle = guard.snapshot()
        guard.restore(handle)
        torch.cuda.synchronize()

    pageable()
    pinned()  # warmup both

    def bench(fn, n=10):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    pg, pn = bench(pageable), bench(pinned)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        f"[IV.7] snapshot+restore  pageable={pg*1e3:.1f}ms  pinned={pn*1e3:.1f}ms  "
        f"speedup={pg/pn:.2f}x  peak_vram={peak_gib:.2f}GiB"
    )
    # Pinned async offload was measured ~3x faster on an RTX PRO 6000; require a
    # clear win with generous headroom so the gate is not noise-sensitive.
    assert pn <= 0.8 * pg
