"""The absorption plan folds a BatchNorm or activation only along an exact single-consumer chain (E271, E272)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.converter import check_representability, convert_torch_model
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityError,
    activation_folds_exactly,
    batchnorm_folds_into,
)

from . import witnesses as W
from .census import CENSUS_MEMBERS, CENSUS_SHAPE, census_defects, traced_plan

CONTRACTS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


def _plan(model, shape, contract):
    return dict(check_representability(model.eval(), shape, packaging=CONTRACTS[contract]).absorption_plan)


class TestFoldExactness:
    @pytest.mark.parametrize("bn, weight, rank, folds", [
        (nn.BatchNorm2d(4), nn.Conv2d(3, 4, 3), 4, True),
        (nn.BatchNorm1d(4), nn.Conv1d(3, 4, 3), 3, True),
        (nn.BatchNorm1d(4), nn.Linear(3, 4), 2, True),
        (ChannelsLastBatchNorm1d(4), nn.Linear(3, 4), 2, True),
        (ChannelsLastBatchNorm1d(4), nn.Linear(3, 4), 3, True),
        (nn.BatchNorm1d(4), nn.Linear(3, 4), 3, False),
        (ChannelsLastBatchNorm1d(4), nn.Conv1d(3, 4, 3), 3, False),
        (nn.BatchNorm2d(4, track_running_stats=False), nn.Conv2d(3, 4, 3), 4, False),
        (nn.BatchNorm2d(4), nn.Conv2d(3, 4, 3), None, False),
    ])
    def test_fold_is_exact_only_on_the_output_channel_axis(self, bn, weight, rank, folds):
        assert batchnorm_folds_into(bn, weight, rank) is folds

    def test_fold_requires_matching_channel_count(self):
        assert batchnorm_folds_into(nn.BatchNorm2d(5), nn.Conv2d(3, 4, 3), 4) is False


class TestActivationFoldExactness:
    """The Perceptron rebuilds an absorbed activation by name, so only default-parameter activations fold."""

    @pytest.mark.parametrize("act, folds", [
        (nn.ReLU(), True), (nn.ReLU(inplace=True), True), (nn.LeakyReLU(), True), (nn.GELU(), True),
        (nn.LeakyReLU(0.2), False), (nn.GELU(approximate="tanh"), False),
    ])
    def test_predicate(self, act, folds):
        assert activation_folds_exactly(act) is folds

    @pytest.mark.parametrize("act", [nn.LeakyReLU(0.2), nn.GELU(approximate="tanh")])
    def test_non_default_activation_stays_a_host_op_and_converts_exactly(self, act):
        class ConvAct(nn.Module):
            def __init__(self):
                super().__init__()
                self.c, self.a, self.fc = nn.Conv2d(3, 8, 3, padding=1), act, nn.Linear(8 * 8 * 8, W.NC)

            def forward(self, x):
                return self.fc(torch.flatten(self.a(self.c(x)), 1))

        assert _plan(ConvAct(), W.SHAPE, "spiking") == {}
        torch.manual_seed(0)
        model = W.fp64_eval_copy(ConvAct())
        assert W.converted_error(model, W.SHAPE, "spiking") <= W.PARITY_BOUND


class TestSingleConsumerPlan:
    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_bn_on_a_fanned_out_weight_output_is_not_folded(self, contract):
        assert "bn" not in _plan(W.FanoutBn(), W.SHAPE, contract)

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_stem_bn_with_a_raw_concat_consumer_is_not_folded(self, contract):
        plan = _plan(W.StemTwoConsumers(), W.SHAPE, contract)
        assert "bn1" not in plan and "bn2" not in plan

    def test_spiking_activation_on_a_fanned_out_weight_output_is_not_folded(self):
        assert _plan(W.FanoutAct(), W.SHAPE, "spiking") == {}

    def test_spiking_bn_after_an_absorbed_activation_is_not_folded(self):
        assert _plan(W.ConvReluBnConv(), W.SHAPE, "spiking") == {
            "a1": "absorbed_into:c1", "a2": "absorbed_into:c2",
        }

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_second_bn_has_no_fold_slot(self, contract):
        plan = _plan(W.DoubleBn(), W.SHAPE, contract)
        assert plan.get("bn1") == "absorbed_into:c" and "bn2" not in plan

    def test_second_spiking_activation_has_no_fold_slot(self):
        assert _plan(W.ActThenAct(), W.SHAPE, "spiking") == {"a1": "absorbed_into:c"}

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_token_axis_bn_after_a_linear_is_not_folded(self, contract):
        assert "bn" not in _plan(W.LinearTokensBn(), (1, 8, 8), contract)

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_channels_last_bn_after_a_token_linear_folds(self, contract):
        assert _plan(W.LinearTokensChannelsLastBn(), (1, 8, 8), contract)["bn"] == "absorbed_into:l1"

    def test_fan_out_after_the_last_chain_member_keeps_the_fold(self):
        class BnFeedsTwo(nn.Module):
            def __init__(self):
                super().__init__()
                self.c, self.bn, self.r = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()
                self.fc = nn.Linear(4 * 8 * 8, W.NC)

            def forward(self, x):
                h = self.bn(self.c(x))
                return self.fc((self.r(h) + h).flatten(1))

        assert _plan(BnFeedsTwo(), W.SHAPE, "mvm") == {"bn": "absorbed_into:c"}
        assert _plan(BnFeedsTwo(), W.SHAPE, "spiking") == {"bn": "absorbed_into:c"}


class TestRunningStatisticsRefusal:
    class _InstanceStatsBn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c, self.bn = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4, track_running_stats=False)
            self.fc = nn.Linear(4 * 8 * 8, W.NC)

        def forward(self, x):
            return self.fc(self.bn(self.c(x)).flatten(1))

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    def test_bn_without_running_statistics_is_refused_by_name(self, contract):
        report = check_representability(self._InstanceStatsBn().eval(), W.SHAPE, packaging=CONTRACTS[contract])
        assert not report.is_representable
        (op,) = report.unsupported_ops
        assert op.node_name == "bn" and op.module_type == "BatchNorm2d"
        assert "running statistics" in (op.reason or "")
        with pytest.raises(RepresentabilityError, match="running statistics"):
            convert_torch_model(self._InstanceStatsBn().eval(), W.SHAPE, W.NC, packaging=CONTRACTS[contract])


class TestCensusOracle:
    """The planner agrees with the census predicate: no fold fans out, defect-free members keep every fold."""

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    @pytest.mark.parametrize("key", sorted(CENSUS_MEMBERS))
    def test_defect_free_members_fold_every_batchnorm(self, key, contract):
        gm, report = traced_plan(CENSUS_MEMBERS[key]().eval(), CENSUS_SHAPE, CONTRACTS[contract])
        assert census_defects(gm, report.absorption_plan) == {"D1": [], "D8": []}

    @pytest.mark.parametrize("contract", ["mvm", "spiking"])
    @pytest.mark.parametrize("key", sorted({**W.D1_WITNESSES, **W.D8_WITNESSES, **W.SPIKING_ONLY_WITNESSES}))
    def test_no_planned_fold_fans_out_on_any_witness(self, key, contract):
        factory, shape = {**W.D1_WITNESSES, **W.D8_WITNESSES, **W.SPIKING_ONLY_WITNESSES}[key]
        gm, report = traced_plan(factory().eval(), shape, CONTRACTS[contract])
        assert census_defects(gm, report.absorption_plan)["D8"] == []
