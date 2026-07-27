"""MVM packaging: bare weight-stationary MMs become packages; activations stay host."""

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import (
    check_representability,
    convert_torch_model,
)


def _convert_mvm(model, input_shape):
    return convert_torch_model(
        model.eval(), input_shape, 4, device="cpu", packaging=MVM_PACKAGING
    )


def _perceptrons(flow):
    return list(flow.get_perceptrons())


def _host_ops(flow):
    return [
        n for n in flow.get_mapper_repr().execution_order()
        if isinstance(n, ComputeOpMapper)
    ]


class TestMvmDetection:
    def test_bare_linear_becomes_a_package(self):
        # Under spiking rules the final Linear is host; under mvm it maps.
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        flow = _convert_mvm(model, (8,))
        ps = _perceptrons(flow)
        assert len(ps) == 2
        assert all(type(p.activation).__name__ == "Identity" for p in ps)
        # The ReLU is the single host op left.
        assert len(_host_ops(flow)) == 1

    def test_bn_is_still_absorbed(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        report = check_representability(model.eval(), (8,), packaging=MVM_PACKAGING)
        assert report.is_representable
        # Absorption plan carries the BN fold only — never the activation.
        assert dict(report.absorption_plan) == {"_1": "absorbed_into:_0"}
        flow = _convert_mvm(model, (8,))
        norms = {type(p.normalization).__name__ for p in _perceptrons(flow)}
        assert "BatchNorm1d" in norms

    def test_conv_without_activation_maps(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 4), nn.ReLU()
        )
        flow = _convert_mvm(model, (1, 8, 8))
        assert len(_perceptrons(flow)) == 2
        assert len(_host_ops(flow)) == 1  # the trailing ReLU

    def test_gelu_stays_on_host(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
        flow = _convert_mvm(model, (8,))
        ps = _perceptrons(flow)
        assert len(ps) == 2
        assert all(type(p.activation).__name__ == "Identity" for p in ps)
        assert len(_host_ops(flow)) == 1

    def test_layernorm_stays_on_host_and_linears_map(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4)
        )
        flow = _convert_mvm(model, (8,))
        assert len(_perceptrons(flow)) == 2
        assert len(_host_ops(flow)) == 1  # LayerNorm

    def test_grouped_conv_is_still_rejected(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 3, groups=2, padding=1), nn.Flatten(),
            nn.Linear(4 * 6 * 6, 4),
        )
        report = check_representability(model.eval(), (4, 6, 6), packaging=MVM_PACKAGING)
        assert not report.is_representable


class TestMvmBoundaries:
    def test_no_encoding_marks_in_the_value_domain(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        flow = _convert_mvm(model, (8,))
        assert all(not p.is_encoding_layer for p in _perceptrons(flow))

    def test_converted_forward_matches_the_native_model(self):
        # Value-domain packaging is numerically the identity regrouping:
        # affine packages + host activations == the native forward.
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 12), nn.GELU(), nn.Linear(12, 4),
        ).eval()
        flow = _convert_mvm(model, (8,)).eval()
        x = torch.randn(5, 8)
        with torch.no_grad():
            torch.testing.assert_close(flow(x), model(x), atol=1e-6, rtol=1e-5)


class TestSpikingDefaultUnchanged:
    def test_default_packaging_is_spiking(self):
        # No packaging argument -> today's behavior (locked in detail by
        # test_packaging_characterization).
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        flow = convert_torch_model(model.eval(), (8,), 4, device="cpu")
        ps = _perceptrons(flow)
        assert len(ps) == 1
        assert ps[0].is_encoding_layer is True
