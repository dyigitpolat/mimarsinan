"""Config-time rules: each fires on its synthetic trigger and stays silent otherwise."""

from mimarsinan.advisories.advisory import SEVERITY_INFO, SEVERITY_UNSUPPORTED
from mimarsinan.advisories.engine import (
    evaluate_config_advisories,
    evaluate_post_pretrain_advisories,
)


def _ids(config):
    return {a.id for a in evaluate_config_advisories(config)}


def _by_id(config, advisory_id):
    fired = [a for a in evaluate_config_advisories(config) if a.id == advisory_id]
    assert len(fired) == 1, f"{advisory_id} fired {len(fired)} times"
    return fired[0]


class TestCascUnsupported:
    _CASC = {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "cascaded"}

    def test_fires_on_cascaded_ttfs(self):
        assert "ADV-CASC-UNSUPPORTED" in _ids(self._CASC)

    def test_default_schedule_is_cascaded(self):
        assert "ADV-CASC-UNSUPPORTED" in _ids({"spiking_mode": "ttfs_cycle_based"})

    def test_silent_on_other_modes(self):
        assert "ADV-CASC-UNSUPPORTED" not in _ids({"spiking_mode": "lif"})
        assert "ADV-CASC-UNSUPPORTED" not in _ids({"spiking_mode": "ttfs"})
        assert "ADV-CASC-UNSUPPORTED" not in _ids({"spiking_mode": "ttfs_quantized"})
        assert "ADV-CASC-UNSUPPORTED" not in _ids({
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
        })

    def test_text_and_severity(self):
        advisory = _by_id(self._CASC, "ADV-CASC-UNSUPPORTED")
        assert advisory.severity == SEVERITY_UNSUPPORTED
        assert "not fully supported" in advisory.detail
        assert "research program" in advisory.detail
        assert "casc_first_crossing_transformation.md" in advisory.detail

    def test_not_a_mandate_violation(self):
        """Cascaded is outside the lif/sync lossless mandate by definition."""
        advisory = _by_id(self._CASC, "ADV-CASC-UNSUPPORTED")
        assert advisory.mandate_violation is False


class TestNovenaCharge:
    _NOVENA = {"spiking_mode": "lif", "firing_mode": "Novena"}

    def test_fires_on_lif_novena(self):
        assert "ADV-NOVENA-CHARGE" in _ids(self._NOVENA)

    def test_silent_on_default_reset_and_non_lif(self):
        assert "ADV-NOVENA-CHARGE" not in _ids({"spiking_mode": "lif"})
        assert "ADV-NOVENA-CHARGE" not in _ids(
            {"spiking_mode": "lif", "firing_mode": "Default"}
        )
        assert "ADV-NOVENA-CHARGE" not in _ids(
            {"spiking_mode": "ttfs", "firing_mode": "TTFS"}
        )

    def test_mandate_violation_on_lif(self):
        advisory = _by_id(self._NOVENA, "ADV-NOVENA-CHARGE")
        assert advisory.mandate_violation is True
        assert advisory.tentative is True
        assert "lif_deployment_exactness.md" in advisory.detail


class TestStrictLtLattice:
    _TRIGGER = {
        "spiking_mode": "lif",
        "thresholding_mode": "<",
        "weight_quantization": True,
    }

    def test_fires_on_strict_lt_with_weight_quantization(self):
        assert "ADV-STRICT-LT-LATTICE" in _ids(self._TRIGGER)

    def test_silent_without_integer_lattice_or_with_inclusive_compare(self):
        assert "ADV-STRICT-LT-LATTICE" not in _ids(
            {"spiking_mode": "lif", "thresholding_mode": "<"}
        )
        assert "ADV-STRICT-LT-LATTICE" not in _ids(
            {
                "spiking_mode": "lif",
                "thresholding_mode": "<=",
                "weight_quantization": True,
            }
        )

    def test_info_severity_and_no_mandate_violation(self):
        advisory = _by_id(self._TRIGGER, "ADV-STRICT-LT-LATTICE")
        assert advisory.severity == SEVERITY_INFO
        assert advisory.mandate_violation is False
        assert "depth_balancing" in advisory.detail
        assert "lif_deployment_exactness.md" in advisory.detail


class TestEnvelopeGate:
    def test_fires_when_pretrain_is_below_the_acceptance_target(self):
        fired = evaluate_post_pretrain_advisories(
            0.90, {"spiking_mode": "lif"}, acceptance_target=0.97
        )
        assert [a.id for a in fired] == ["ADV-ENVELOPE-GATE"]
        assert "mixer_column_scale_pathology.md" in fired[0].detail

    def test_silent_at_or_above_the_target(self):
        assert evaluate_post_pretrain_advisories(
            0.97, {}, acceptance_target=0.97
        ) == []
        assert evaluate_post_pretrain_advisories(
            0.99, {}, acceptance_target=0.97
        ) == []

    def test_silent_when_the_run_declares_no_target(self):
        assert evaluate_post_pretrain_advisories(0.10, {}) == []

    def test_config_fallback_reads_target_metric_override(self):
        fired = evaluate_post_pretrain_advisories(
            0.5, {"target_metric_override": 0.9}
        )
        assert [a.id for a in fired] == ["ADV-ENVELOPE-GATE"]

    def test_not_a_mandate_violation_even_for_lif(self):
        """An envelope deficit is pretrain-side: lossless deployment refinement
        cannot raise a float envelope, so the refinement checklist must not own it."""
        fired = evaluate_post_pretrain_advisories(
            0.5, {"spiking_mode": "lif"}, acceptance_target=0.9
        )
        assert fired[0].mandate_violation is False
