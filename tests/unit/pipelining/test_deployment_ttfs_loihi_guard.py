import pytest

from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs


def test_loihi_sim_with_ttfs_raises_at_assembly():
    with pytest.raises(ValueError, match="enable_loihi_simulation"):
        get_pipeline_step_specs({
            "spiking_mode": "ttfs_quantized",
            "enable_loihi_simulation": True,
            "enable_sanafe_simulation": False,
            "weight_quantization": True,
            "activation_quantization": True,
        })
