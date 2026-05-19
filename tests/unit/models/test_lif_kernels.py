import torch

from mimarsinan.models.lif_kernels import lif_fire_and_reset


def test_lif_fire_and_reset_default_mode():
    memb = torch.tensor([[1.5]])
    th = torch.tensor(1.0)
    spikes = lif_fire_and_reset(
        memb, th, thresholding_mode="<=", firing_mode="Default"
    )
    assert spikes.item() == 1.0
    assert memb.item() == 0.5
