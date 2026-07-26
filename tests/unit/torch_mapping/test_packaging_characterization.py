"""Characterization lock: today's perceptron-packaging decisions, pinned.

The W1 packaging-contract refactor parameterizes detection/absorption; these
pins prove the spiking contract path reproduces today's structure exactly.
"""

import torch.nn as nn

from mimarsinan.torch_mapping.converter import (
    check_representability,
    convert_torch_model,
)


def _structure(flow):
    """Deterministic structural signature of a converted flow."""
    rows = []
    for node in flow.get_mapper_repr().execution_order():
        row = type(node).__name__
        for group in getattr(node, "owned_perceptron_groups", lambda: [])():
            for p in group:
                row += (
                    f"[act={type(p.activation).__name__}"
                    f",norm={type(p.normalization).__name__}"
                    f",enc={bool(p.is_encoding_layer)}]"
                )
        rows.append(row)
    return rows


def _convert(model, input_shape, **kwargs):
    return convert_torch_model(model.eval(), input_shape, 4, device="cpu", **kwargs)


class TestAbsorptionPlanPins:
    def test_linear_bn_relu_chain_absorbs_into_linear(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        report = check_representability(model.eval(), (8,))
        assert report.is_representable
        assert dict(report.absorption_plan) == {
            "_1": "absorbed_into:_0",
            "_2": "absorbed_into:_0",
        }

    def test_gelu_absorbs_like_relu(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4), nn.ReLU())
        report = check_representability(model.eval(), (8,))
        assert dict(report.absorption_plan) == {
            "_1": "absorbed_into:_0",
            "_3": "absorbed_into:_2",
        }

    def test_layernorm_is_never_absorbed(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4), nn.ReLU()
        )
        report = check_representability(model.eval(), (8,))
        assert dict(report.absorption_plan) == {"_3": "absorbed_into:_2"}

    def test_grouped_conv_is_unsupported(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 3, groups=2, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 6 * 6, 4), nn.ReLU(),
        )
        report = check_representability(model.eval(), (4, 6, 6))
        assert not report.is_representable
        assert any("Grouped convolution" in (op.reason or "")
                   for op in report.unsupported_ops)


class TestConvertedStructurePins:
    """The mappable-vs-host decision, pinned node-by-node (spiking contract)."""

    def test_mlp_with_bare_final_linear(self):
        # MM+BN+ReLU packages; the bare final Linear stays a host ComputeOp.
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        assert _structure(_convert(model, (8,))) == [
            "InputMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=BatchNorm1d,enc=True]",
            "Ensure2DMapper",
            "ComputeOpMapper",
        ]

    def test_gelu_perceptron_keeps_gelu_activation(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4), nn.ReLU())
        assert _structure(_convert(model, (8,))) == [
            "InputMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=GELU,norm=Identity,enc=True]",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=Identity,enc=False]",
        ]

    def test_conv_bn_relu_then_linear(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 6 * 6, 4), nn.ReLU(),
        )
        assert _structure(_convert(model, (1, 8, 8))) == [
            "InputMapper",
            "Conv2DPerceptronMapper[act=LeakyGradReLU,norm=BatchNorm1d,enc=True]",
            "ReshapeMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=Identity,enc=False]",
        ]

    def test_conv_without_activation_stays_host(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 4), nn.ReLU()
        )
        assert _structure(_convert(model, (1, 8, 8))) == [
            "InputMapper",
            "ComputeOpMapper",
            "ReshapeMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=Identity,enc=True]",
        ]

    def test_bare_linear_and_layernorm_run_on_host(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4), nn.ReLU()
        )
        assert _structure(_convert(model, (8,))) == [
            "InputMapper",
            "Ensure2DMapper",
            "ComputeOpMapper",
            "ComputeOpMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=Identity,enc=False]",
        ]

    def test_offload_placement_clears_encoding_mark(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
        )
        flow = _convert(model, (8,), encoding_layer_placement="offload")
        assert _structure(flow) == [
            "InputMapper",
            "Ensure2DMapper",
            "PerceptronMapper[act=LeakyGradReLU,norm=BatchNorm1d,enc=False]",
            "Ensure2DMapper",
            "ComputeOpMapper",
        ]
