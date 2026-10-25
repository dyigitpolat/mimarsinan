"""[fix_b] x_fluid and R read one stationary weight count per network, the same on every target.

The logical weight cells were counted from the layout's specs: a bank-backed spec once per bank, every other spec
once per spec. A transformer's token-wise linear layers lay out one spec per token position without a shared bank on
some geometries and with one on others, so the "logical" count of one ViT ranged 130 x across the shipped targets
(516,096 cells on TrueNorth, 67,365,898 on ISAAC-like), and x_fluid and R (MACs per weight) were not properties of the
network. The stationary count is the converted model's weights: per on-chip perceptron, its weight tensor once plus
one bias row per output (the cells the program writes once per weight bank, whatever the geometry). The per-position
copies are the replication factor (as-mapped cells over stationary cells).
"""

from __future__ import annotations

import contextlib
import io
import warnings

import pytest
import torch
import torch.nn as nn

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


class _TokenMixer(nn.Module):
    """Four tokens of eight features: a token-wise Linear 8 -> 6 (one weight bank applied at every token), ReLU, then
    a head over the flattened tokens."""

    def __init__(self):
        super().__init__()
        self.token, self.relu, self.head = nn.Linear(8, 6), nn.ReLU(), nn.Linear(24, 3)

    def forward(self, x):
        return self.head(torch.flatten(self.relu(self.token(x)), 1))


class _Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c, self.r, self.pool, self.fc = nn.Conv2d(2, 5, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), \
            nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.r(self.c(x))), 1))


def _network(model: nn.Module, shape) -> ladder.LadderNetwork:
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model.eval(), shape, 3, packaging=MVM_PACKAGING, strict=False).eval()
    return ladder.LadderNetwork(name=type(model).__name__, mapper_repr=flow.get_mapper_repr(), resolution=shape)


def _platform(axons: int, neurons: int, has_bias: bool) -> dict:
    return {"cores": [{"max_axons": axons, "max_neurons": neurons, "count": 64, "has_bias": has_bias}],
            "allow_scheduling": True, "allow_coalescing": True, "allow_neuron_splitting": True, "weight_bits": 8}


GEOMETRIES = [(16, 4, False), (64, 64, True), (256, 256, False)]
VEHICLES = [(_TokenMixer, (4, 8), (8 + 1) * 6 + (24 + 1) * 3), (_Conv, (2, 6, 6), (2 * 9 + 1) * 5 + (5 + 1) * 3)]


class TestOneStationaryCount:
    @pytest.mark.parametrize("model, shape, cells", VEHICLES, ids=["token_mixer", "conv"])
    def test_the_count_is_the_models_weights_plus_one_bias_row_on_every_geometry(self, model, shape, cells):
        network = _network(model(), shape)
        counts = {g: ladder.shape_census(network, _platform(*g)).logical_cells for g in GEOMETRIES}
        assert set(counts.values()) == {cells}, counts

    def test_the_token_copies_are_replication(self):
        network = _network(_TokenMixer(), (4, 8))
        census = ladder.shape_census(network, _platform(64, 64, True))
        assert census.as_mapped_cells >= 4 * 8 * 6
        assert census.as_mapped_cells / census.logical_cells > 1.0
