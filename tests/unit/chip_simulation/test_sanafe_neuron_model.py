"""SANA-FE neuron model attribute mapping."""

from mimarsinan.chip_simulation.sanafe.neuron_model import lif_model_attributes


def test_lif_model_attributes_novena_reset_mode():
    attrs = lif_model_attributes(threshold=1.0, firing_mode="Novena")
    assert attrs["reset_mode"] == "hard"


def test_lif_model_attributes_default_reset_mode():
    attrs = lif_model_attributes(threshold=1.0, firing_mode="Default")
    assert attrs["reset_mode"] == "soft"
