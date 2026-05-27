import torch

from mimarsinan.models.ttfs_activation import ttfs_activation_from_type


def test_ttfs_activation_relu_default():
    fn = ttfs_activation_from_type(None)
    x = torch.tensor([-1.0, 0.5, 2.0])
    assert torch.allclose(fn(x), torch.relu(x))


def test_ttfs_activation_compound_string():
    fn = ttfs_activation_from_type("GELU + extra")
    x = torch.tensor([0.0, 1.0])
    assert fn(x).shape == x.shape


def test_ttfs_activation_identity():
    fn = ttfs_activation_from_type("Identity")
    x = torch.tensor([-1.0, 2.0])
    assert torch.allclose(fn(x), x)
