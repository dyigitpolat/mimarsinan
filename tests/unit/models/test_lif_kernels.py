"""lif_fire_and_reset: sync-free masking, bit-exact to the fancy-index reference."""

import pytest
import torch

from mimarsinan.models.nn.lif_kernels import lif_fire_and_reset

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def _reference_fire_and_reset(memb, threshold, *, thresholding_mode, firing_mode,
                              output_dtype=None):
    """The historical boolean-fancy-index implementation (forces a GPU sync)."""
    fired = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    if firing_mode == "Novena":
        memb[fired] = 0.0
    elif firing_mode == "Default":
        memb[fired] -= threshold
    if output_dtype is not None:
        return fired.to(output_dtype)
    return fired.float()


def test_lif_fire_and_reset_default_mode():
    memb = torch.tensor([[1.5]])
    th = torch.tensor(1.0)
    spikes = lif_fire_and_reset(
        memb, th, thresholding_mode="<=", firing_mode="Default"
    )
    assert spikes.item() == 1.0
    assert memb.item() == 0.5


class TestBitExactAgainstFancyIndexReference:
    @pytest.mark.parametrize("firing_mode", ["Default", "Novena"])
    @pytest.mark.parametrize("thresholding_mode", ["<", "<="])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_randomized_membranes(self, firing_mode, thresholding_mode, dtype):
        gen = torch.Generator().manual_seed(3)
        memb = torch.randn(16, 32, generator=gen, dtype=dtype) * 2.0
        th = torch.tensor(0.75, dtype=dtype)
        memb_ref = memb.clone()

        got = lif_fire_and_reset(
            memb, th, thresholding_mode=thresholding_mode,
            firing_mode=firing_mode,
        )
        want = _reference_fire_and_reset(
            memb_ref, th, thresholding_mode=thresholding_mode,
            firing_mode=firing_mode,
        )
        assert torch.equal(got, want)
        # Bitwise membrane equality, signed zeros included.
        int_view = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(memb.view(int_view), memb_ref.view(int_view))

    @pytest.mark.parametrize("firing_mode", ["Default", "Novena"])
    def test_exact_boundary_values(self, firing_mode):
        # memb == threshold exercises the <= vs < edge exactly.
        th = torch.tensor(1.0, dtype=torch.float64)
        memb = torch.tensor(
            [[1.0, 1.0 - 2**-52, 1.0 + 2**-52, -1.0, 0.0]], dtype=torch.float64,
        )
        for mode in ("<", "<="):
            m1, m2 = memb.clone(), memb.clone()
            got = lif_fire_and_reset(
                m1, th, thresholding_mode=mode, firing_mode=firing_mode,
            )
            want = _reference_fire_and_reset(
                m2, th, thresholding_mode=mode, firing_mode=firing_mode,
            )
            assert torch.equal(got, want)
            assert torch.equal(m1.view(torch.int64), m2.view(torch.int64))

    def test_output_dtype_honored(self):
        memb = torch.tensor([[2.0, 0.1]])
        th = torch.tensor(1.0)
        out = lif_fire_and_reset(
            memb, th, thresholding_mode="<=", firing_mode="Default",
            output_dtype=torch.float64,
        )
        assert out.dtype == torch.float64
        assert out.tolist() == [[1.0, 0.0]]

    def test_no_fancy_index_synchronization_path(self):
        """The reset must not use boolean fancy indexing (nonzero forces a
        host-device sync per call — the t0_01 profile's 189 s hot spot)."""
        import inspect

        import mimarsinan.models.nn.lif_kernels as mod

        src = inspect.getsource(mod.lif_fire_and_reset)
        assert "memb[fired] =" not in src
        assert "memb[fired] -=" not in src
