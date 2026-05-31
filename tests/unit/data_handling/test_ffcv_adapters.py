"""Unit tests for the FFCV→torch shim and the GPU post-decode stage."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.data_handling.ffcv.adapters import GPUResizeNormalize, TorchLoaderShim


class _FakeFFCVLoader:
    """Minimal stand-in: yields uint8 H×W×C-style batches like FFCV does."""

    def __init__(self, n_batches, *, batch=4, h=32, w=32, c=3, channels_last=False):
        self.n = n_batches
        self.batch = batch
        self.h, self.w, self.c = h, w, c
        self.channels_last = channels_last
        self.batch_size = batch
        self.num_workers = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            if self.channels_last:
                x = torch.full((self.batch, self.h, self.w, self.c), 128, dtype=torch.uint8)
            else:
                x = torch.full((self.batch, self.c, self.h, self.w), 128, dtype=torch.uint8)
            y = torch.zeros(self.batch, dtype=torch.long)
            yield x, y


class TestGPUResizeNormalize:
    def test_uint8_chw_to_normalized_chw(self):
        op = GPUResizeNormalize(resize_to=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        x = torch.full((2, 3, 4, 4), 128, dtype=torch.uint8)
        y = op.apply(x)
        # (128/255 - 0.5) / 0.5 ≈ 0.00392... → near zero
        assert y.dtype == torch.float32
        assert y.shape == (2, 3, 4, 4)
        assert torch.allclose(y, torch.tensor(0.00392).expand_as(y), atol=1e-3)

    def test_channels_last_promotion(self):
        op = GPUResizeNormalize(resize_to=None)
        x = torch.full((2, 4, 4, 3), 128, dtype=torch.uint8)
        y = op.apply(x)
        assert y.shape == (2, 3, 4, 4)

    def test_resize(self):
        op = GPUResizeNormalize(resize_to=16, mean=(0, 0, 0), std=(1, 1, 1))
        x = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
        y = op.apply(x)
        assert y.shape == (1, 3, 16, 16)


class TestGPUResizeAndNormalizeSplit:
    """The split pair (GPUResize + GPUNormalize) reproduces GPUResizeNormalize exactly.

    Splitting these is what lets us slot a kornia augment in the middle, where
    pixel values are still in the [0, 1] range that kornia's pixel-level ops
    (Posterize, Solarize, Sharpness, ...) assume.
    """

    def test_resize_only_emits_unnormalized_floats(self):
        from mimarsinan.data_handling.ffcv.adapters import GPUResize
        op = GPUResize(resize_to=None, scale_255=True)
        x = torch.full((2, 3, 4, 4), 128, dtype=torch.uint8)
        y = op.apply(x)
        assert y.dtype == torch.float32
        assert y.shape == (2, 3, 4, 4)
        # 128/255 ≈ 0.502
        assert torch.allclose(y, torch.full_like(y, 128 / 255.0), atol=1e-6)

    def test_normalize_only_applies_mean_std(self):
        from mimarsinan.data_handling.ffcv.adapters import GPUNormalize
        op = GPUNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        x = torch.full((1, 3, 4, 4), 0.5)
        y = op.apply(x)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_split_chain_equals_composite(self):
        """GPUResize + GPUNormalize ≡ GPUResizeNormalize for the same params."""
        from mimarsinan.data_handling.ffcv.adapters import (
            GPUNormalize,
            GPUResize,
            GPUResizeNormalize,
        )
        x = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        composite = GPUResizeNormalize(
            resize_to=None, mean=mean, std=std, scale_255=True
        ).apply(x)

        split = GPUNormalize(mean=mean, std=std).apply(
            GPUResize(resize_to=None, scale_255=True).apply(x)
        )
        assert torch.allclose(composite, split, atol=1e-6), \
            "GPUResize+GPUNormalize must reproduce GPUResizeNormalize exactly"

    def test_split_chain_with_resize(self):
        from mimarsinan.data_handling.ffcv.adapters import (
            GPUNormalize,
            GPUResize,
            GPUResizeNormalize,
        )
        x = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        composite = GPUResizeNormalize(
            resize_to=64, mean=mean, std=std, scale_255=True, interpolation="bicubic"
        ).apply(x)
        split = GPUNormalize(mean=mean, std=std).apply(
            GPUResize(resize_to=64, scale_255=True, interpolation="bicubic").apply(x)
        )
        assert torch.allclose(composite, split, atol=1e-6)

    def test_split_grayscale_reduction_stays_in_resize(self):
        """to_grayscale (channel averaging) belongs in GPUResize (operates in [0,1])."""
        from mimarsinan.data_handling.ffcv.adapters import GPUResize
        op = GPUResize(resize_to=None, scale_255=True, to_grayscale=True)
        x = torch.full((1, 3, 4, 4), 128, dtype=torch.uint8)
        y = op.apply(x)
        assert y.shape == (1, 1, 4, 4)


class TestTorchLoaderShim:
    def test_iter_yields_post_processed_batches(self):
        inner = _FakeFFCVLoader(n_batches=3)
        op = GPUResizeNormalize(resize_to=None, mean=(0, 0, 0), std=(1, 1, 1))
        shim = TorchLoaderShim(inner, postprocess=op)
        batches = list(shim)
        assert len(batches) == 3
        x, y = batches[0]
        assert x.dtype == torch.float32  # normalized
        assert y.shape == (4,)
        assert y.dtype == torch.long

    def test_iter_without_postprocess_passes_through(self):
        inner = _FakeFFCVLoader(n_batches=2)
        shim = TorchLoaderShim(inner)
        batches = list(shim)
        assert len(batches) == 2
        x, y = batches[0]
        assert x.dtype == torch.uint8  # untouched

    def test_len(self):
        inner = _FakeFFCVLoader(n_batches=7)
        shim = TorchLoaderShim(inner)
        assert len(shim) == 7

    def test_attrs_match_torch_loader_surface(self):
        inner = _FakeFFCVLoader(n_batches=1, batch=16)
        shim = TorchLoaderShim(inner)
        assert shim.batch_size == 16
        assert shim.num_workers == 0

    def test_iter_isolates_from_rotating_buffer_pool(self):
        """The shim must clone so downstream consumers can hold references safely.

        FFCV's Loader yields tensor views into a fixed rotating pool of buffers
        and overwrites them as iteration advances. Any consumer that holds the
        yielded tensor across iteration boundaries (val_cache, test_on_subsample,
        simulation_runner, test_sample_loader, ...) used to silently see
        corrupted data. The shim is the single boundary that owns the FFCV
        contract; cloning here keeps all downstream code safe by construction.
        """
        # Loader that reuses 2 buffers and writes a distinct value per batch.
        bs, c, h, w = 4, 3, 4, 4
        x_pool = [torch.empty((bs, c, h, w), dtype=torch.uint8) for _ in range(2)]
        y_pool = [torch.empty((bs,), dtype=torch.long) for _ in range(2)]
        per_batch = [torch.full((bs, c, h, w), i + 1, dtype=torch.uint8) for i in range(4)]
        per_batch_labels = [torch.full((bs,), i + 1, dtype=torch.long) for i in range(4)]

        class _PoolLoader:
            batch_size = bs
            num_workers = 0
            def __len__(self): return 4
            def __iter__(self):
                for i in range(4):
                    slot = i % 2
                    x_pool[slot].copy_(per_batch[i])
                    y_pool[slot].copy_(per_batch_labels[i])
                    yield x_pool[slot], y_pool[slot]

        # Without postprocess (worst case — no _post.apply() to obscure aliasing).
        shim = TorchLoaderShim(_PoolLoader())
        held = list(shim)

        # Overwrite the pool with garbage; downstream references must survive.
        for slot in range(2):
            x_pool[slot].fill_(0xFF)
            y_pool[slot].fill_(-1)

        for i, (x, y) in enumerate(held):
            assert torch.equal(x, per_batch[i]), \
                f"shim leaked rotating-buffer reference at batch {i}"
            assert torch.equal(y, per_batch_labels[i]), \
                f"shim leaked rotating-label reference at batch {i}"

    def test_postprocess_chain_applies_in_order(self):
        """Multiple gpu_postprocess ops chain in declared order."""
        inner = _FakeFFCVLoader(n_batches=1, batch=2, h=4, w=4, c=3)

        class _AddOne:
            def apply(self, x):
                return x.float() + 1.0

        class _Double:
            def apply(self, x):
                return x.float() * 2.0

        shim = TorchLoaderShim(inner, postprocess=[_AddOne(), _Double()])
        x, _ = next(iter(shim))
        # baseline x is 128 (uint8) → after AddOne→129 → after Double→258
        assert torch.allclose(x, torch.full_like(x, 258.0))


    def test_postprocess_accepts_single_op_for_back_compat(self):
        inner = _FakeFFCVLoader(n_batches=1, batch=2, h=4, w=4, c=3)
        op = GPUResizeNormalize(resize_to=None, mean=(0, 0, 0), std=(1, 1, 1))
        shim = TorchLoaderShim(inner, postprocess=op)
        x, _ = next(iter(shim))
        assert x.dtype == torch.float32

    def test_iter_isolates_postprocessed_batches_too(self):
        """Even with a postprocess that materializes x, the label y must clone."""
        bs, c, h, w = 4, 3, 4, 4
        x_pool = [torch.empty((bs, c, h, w), dtype=torch.uint8) for _ in range(2)]
        y_pool = [torch.empty((bs,), dtype=torch.long) for _ in range(2)]
        labels = [torch.full((bs,), i + 1, dtype=torch.long) for i in range(4)]

        class _PoolLoader:
            batch_size = bs
            num_workers = 0
            def __len__(self): return 4
            def __iter__(self):
                for i in range(4):
                    slot = i % 2
                    x_pool[slot].fill_(i + 1)
                    y_pool[slot].copy_(labels[i])
                    yield x_pool[slot], y_pool[slot]

        op = GPUResizeNormalize(resize_to=None, mean=(0, 0, 0), std=(1, 1, 1))
        shim = TorchLoaderShim(_PoolLoader(), postprocess=op)
        held = list(shim)

        for slot in range(2):
            y_pool[slot].fill_(-1)

        for i, (_x, y) in enumerate(held):
            assert torch.equal(y, labels[i]), f"shim leaked label reference at batch {i}"
