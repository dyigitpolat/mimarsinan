"""Advisory engine contract: stable ids, legal severities, machine-readable payloads."""

import dataclasses

import pytest

from mimarsinan.advisories.advisory import (
    SEVERITIES,
    SEVERITY_INFO,
    SEVERITY_RISK,
    SEVERITY_UNSUPPORTED,
    Advisory,
    lossless_mandate_applies,
)
from mimarsinan.advisories.engine import (
    ALL_ADVISORY_IDS,
    evaluate_config_advisories,
    evaluate_post_pretrain_advisories,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

EXPECTED_IDS = {
    "ADV-CASC-UNSUPPORTED",
    "ADV-STAIRCASE-DEPTH",
    "ADV-SCALE-SPREAD",
    "ADV-NORMFREE-CHAIN",
    "ADV-BIAS-GRID-DOMINANCE",
    "ADV-FANIN-DEPTH-IMBALANCE",
    "ADV-NOVENA-CHARGE",
    "ADV-STRICT-LT-LATTICE",
    "ADV-ENVELOPE-GATE",
}


class TestAdvisoryDataclass:
    def _advisory(self, **over):
        kwargs = dict(
            id="ADV-TEST",
            severity=SEVERITY_RISK,
            title="t",
            detail="d",
            tentative=True,
            mandate_violation=False,
            suggested_levers=("lever_a",),
        )
        kwargs.update(over)
        return Advisory(**kwargs)

    def test_frozen(self):
        advisory = self._advisory()
        with pytest.raises(dataclasses.FrozenInstanceError):
            advisory.id = "other"  # type: ignore[misc]

    def test_unknown_severity_rejected(self):
        with pytest.raises(ValueError):
            self._advisory(severity="WARNING")

    def test_severity_taxonomy(self):
        assert SEVERITIES == {SEVERITY_UNSUPPORTED, SEVERITY_RISK, SEVERITY_INFO}

    def test_payload_is_machine_readable(self):
        payload = self._advisory().as_payload()
        assert payload == {
            "id": "ADV-TEST",
            "severity": SEVERITY_RISK,
            "title": "t",
            "detail": "d",
            "tentative": True,
            "mandate_violation": False,
            "suggested_levers": ["lever_a"],
        }


class TestStableIds:
    def test_engine_exposes_exactly_the_rule_set(self):
        assert ALL_ADVISORY_IDS == frozenset(EXPECTED_IDS)

    def test_fired_advisories_use_registered_ids(self):
        config = {
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "cascaded",
            "firing_mode": "TTFS",
        }
        for advisory in evaluate_config_advisories(config):
            assert advisory.id in ALL_ADVISORY_IDS
        for advisory in evaluate_post_pretrain_advisories(
            0.1, {}, acceptance_target=0.9
        ):
            assert advisory.id in ALL_ADVISORY_IDS


def _plan(config):
    return DeploymentPlan.resolve(config)


class TestLosslessMandatePredicate:
    def test_lif_and_sync_are_in_mandate_scope(self):
        assert lossless_mandate_applies(_plan({"spiking_mode": "lif"}))
        assert lossless_mandate_applies(_plan({
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
        }))

    def test_cascaded_and_analytic_ttfs_are_not(self):
        assert not lossless_mandate_applies(_plan({
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "cascaded",
        }))
        assert not lossless_mandate_applies(_plan({"spiking_mode": "ttfs"}))
        assert not lossless_mandate_applies(
            _plan({"spiking_mode": "ttfs_quantized"})
        )


class TestPlanOrConfigDuckTyping:
    def test_accepts_a_resolved_deployment_plan(self):
        plan = _plan({
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "cascaded",
        })
        ids = [a.id for a in evaluate_config_advisories(plan)]
        assert "ADV-CASC-UNSUPPORTED" in ids

    def test_accepts_a_raw_config_dict(self):
        ids = [
            a.id
            for a in evaluate_config_advisories(
                {"spiking_mode": "ttfs_cycle_based"}
            )
        ]
        assert "ADV-CASC-UNSUPPORTED" in ids
