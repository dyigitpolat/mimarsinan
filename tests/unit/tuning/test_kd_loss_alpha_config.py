"""The KD-blend fine-tune loss weighting (CE alpha / KD temperature) is config-driven.

Both LIF and TTFS conversion fine-tuning share ``KDClassificationLoss`` =
``alpha*CE + (1-alpha)*KD`` via ``KDBlendAdaptationTuner``. The historical hardcode
is alpha=0.3 / T=3.0 (KD-heavy). These were not configurable, so a KD-heavy objective
could not be re-weighted toward hard-label CE for harder datasets where it under-fits.
This locks the new ``kd_ce_alpha`` / ``kd_temperature`` knobs: default-preserving
(0.3 / 3.0 == the historical hardcode) and threaded into the loss when set.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS as DEFAULTS,
    CONFIG_KEYS_SET,
)
from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)


def _stub_tuner(config):
    """A minimal object exposing exactly what ``_kd_classification_loss`` reads."""
    return SimpleNamespace(
        pipeline=SimpleNamespace(config=config),
        _teacher=nn.Linear(4, 3),
    )


class TestDefaultsArePreserved:
    def test_default_loss_is_the_historical_hardcode(self):
        loss = KDClassificationLoss(nn.Linear(4, 3))
        assert loss.alpha == 0.3
        assert loss.temperature == 3.0

    def test_config_defaults_registered(self):
        assert DEFAULTS["kd_ce_alpha"] == 0.3
        assert DEFAULTS["kd_temperature"] == 3.0
        assert "kd_ce_alpha" in CONFIG_KEYS_SET
        assert "kd_temperature" in CONFIG_KEYS_SET

    def test_tuner_builds_default_loss_when_keys_absent(self):
        loss = KDBlendAdaptationTuner._kd_classification_loss(
            _stub_tuner({}), nn.Linear(4, 3)
        )
        assert loss.alpha == 0.3       # byte-identical to the old hardcode
        assert loss.temperature == 3.0


class TestConfigThreadsThrough:
    def test_tuner_reads_alpha_and_temperature_from_config(self):
        loss = KDBlendAdaptationTuner._kd_classification_loss(
            _stub_tuner({"kd_ce_alpha": 0.8, "kd_temperature": 2.0}),
            nn.Linear(4, 3),
        )
        assert loss.alpha == 0.8
        assert loss.temperature == 2.0

    def test_ce_dominant_alpha_one_is_pure_cross_entropy(self):
        torch.manual_seed(0)
        teacher = nn.Linear(4, 3)
        model = nn.Linear(4, 3)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss_fn = KDClassificationLoss(teacher, temperature=3.0, alpha=1.0)
        got = loss_fn(model, x, y)
        ce = torch.nn.functional.cross_entropy(model(x), y)
        assert torch.allclose(got, ce, atol=1e-6)  # alpha=1 => KD term drops out

    def test_alpha_blends_ce_and_kd(self):
        torch.manual_seed(1)
        teacher = nn.Linear(4, 3)
        model = nn.Linear(4, 3)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        a = 0.6
        blended = KDClassificationLoss(teacher, temperature=3.0, alpha=a)(model, x, y)
        ce_only = KDClassificationLoss(teacher, temperature=3.0, alpha=1.0)(model, x, y)
        kd_only = KDClassificationLoss(teacher, temperature=3.0, alpha=0.0)(model, x, y)
        assert torch.allclose(blended, a * ce_only + (1.0 - a) * kd_only, atol=1e-6)
