"""SkipPerceptronMixer construction and forward contract."""

import torch

from mimarsinan.models.perceptron_mixer.skip_perceptron_mixer import SkipPerceptronMixer


def _tiny_mixer():
    return SkipPerceptronMixer(
        "cpu",
        input_shape=(1, 4, 4),
        num_classes=3,
        patch_n_1=2,
        patch_m_1=2,
        patch_c_1=3,
        fc_w_1=8,
        fc_k_1=2,
        patch_n_2=2,
        patch_c_2=2,
        fc_w_2=6,
        fc_k_2=2,
    )


def test_constructs_with_device_like_sibling_flows():
    model = _tiny_mixer()
    assert model.device == "cpu"


def test_forward_output_shape():
    model = _tiny_mixer().eval()
    x = torch.randn(4, 1, 4, 4)
    out = model(x)
    assert out.shape == (4, 3)


def test_get_perceptrons_covers_all_layers():
    model = _tiny_mixer()
    perceptrons = model.get_perceptrons()
    expected = (
        model.patch_count
        + model.fc_depth
        + model.patch_count_2
        + model.fc_depth_2
        + 1
    )
    assert len(perceptrons) == expected
