import torch

from mimarsinan.models.ttfs_kernels import ttfs_quantized_activation


def test_ttfs_quantized_saturates_at_one():
    V = torch.tensor([[1.0]])
    th = torch.tensor([[1.0]])
    out = ttfs_quantized_activation(V, th, simulation_length=4)
    assert out.item() == 1.0
