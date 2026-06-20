"""Optimization-driver resolution (P3 of the trifecta).

``OptimizationDriver.resolve`` maps the (already precedence-resolved) fast-path
selectors to the concrete ``_setup_fast_ladder`` rung — controller (default) vs the
fixed-ladder fast driver. The STE-fast variant owns a single rung at rate 1.0; the
blend/proxy variants walk the multi-rung ladder, with the proxy flooring the endpoint
LR. The rung derivation lives (and is tested) in one place.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.optimization_driver import OptimizationDriver

_BLEND_RATES = [0.5, 0.75, 0.9, 0.97, 1.0]


def _resolve(
    *,
    staircase_ste_fast=False,
    genuine_blend_fast=False,
    proxy_fast=False,
    ste_steps=1000,
    blend_fast_rates=None,
    blend_fast_steps_per_rate=120,
    blend_fast_lr_eta_min=0.1,
):
    return OptimizationDriver.resolve(
        staircase_ste_fast=staircase_ste_fast,
        genuine_blend_fast=genuine_blend_fast,
        proxy_fast=proxy_fast,
        ste_steps=ste_steps,
        blend_fast_rates=blend_fast_rates if blend_fast_rates is not None else _BLEND_RATES,
        blend_fast_steps_per_rate=blend_fast_steps_per_rate,
        blend_fast_lr_eta_min=blend_fast_lr_eta_min,
    )


class TestController:
    def test_default_is_controller(self):
        d = _resolve()
        assert d.controller is True
        assert d.fast_ladder is False


class TestSteFast:
    def test_single_rung_at_full_rate(self):
        d = _resolve(staircase_ste_fast=True, ste_steps=777)
        assert d.fast_ladder is True
        assert d.fast_ladder_rates == [1.0]
        assert d.fast_ladder_steps_per_rate == 777
        assert d.fast_ladder_eta_min_factor == 0.0

    def test_ste_fast_wins_over_blend_proxy_selectors(self):
        # The plan never sets these together, but the driver resolution must be
        # unambiguous: STE-fast takes precedence (its dedicated loop).
        d = _resolve(staircase_ste_fast=True, proxy_fast=True, genuine_blend_fast=True)
        assert d.fast_ladder_rates == [1.0]


class TestBlendFast:
    def test_multi_rung_no_eta_floor(self):
        d = _resolve(genuine_blend_fast=True)
        assert d.fast_ladder is True
        assert d.fast_ladder_rates == _BLEND_RATES
        assert d.fast_ladder_eta_min_factor == 0.0


class TestProxyFast:
    def test_proxy_floors_endpoint_lr(self):
        d = _resolve(proxy_fast=True, blend_fast_lr_eta_min=0.25)
        assert d.fast_ladder is True
        assert d.fast_ladder_rates == _BLEND_RATES
        assert d.fast_ladder_eta_min_factor == 0.25

    def test_proxy_steps_per_rate_carried(self):
        d = _resolve(proxy_fast=True, blend_fast_steps_per_rate=99)
        assert d.fast_ladder_steps_per_rate == 99
