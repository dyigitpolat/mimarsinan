"""The wizard resolve payload carries config-time deployment advisories."""

from pathlib import Path

from mimarsinan.gui.wizard.schema_api import resolve_payload

_WIZARD_HTML = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "wizard.html"
)


def _minimal_draft(**parts) -> dict:
    draft = {
        "experiment_name": "advisory_wizard_test",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": "./generated",
        "start_step": None,
        "deployment_parameters": {},
        "platform_constraints": {},
    }
    draft.update(parts)
    dp = draft["deployment_parameters"]
    dp.setdefault("model_type", "simple_mlp")
    dp.setdefault("model_config", {})
    return draft


class TestWizardAdvisories:
    def test_cascaded_pick_shows_the_unsupported_warning(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={"spiking_family": "ttfs", "spiking_variant": "cascaded"}
        ))
        assert payload["ok"], payload["errors"]
        by_id = {row["id"]: row for row in payload["advisories"]}
        assert "ADV-CASC-UNSUPPORTED" in by_id
        row = by_id["ADV-CASC-UNSUPPORTED"]
        assert row["severity"] == "UNSUPPORTED"
        assert "not fully supported" in row["detail"]
        assert isinstance(row["suggested_levers"], list)

    def test_streamed_pick_shows_the_structural_contract_info(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={
                "spiking_family": "lif", "spiking_variant": "streamed",
            }
        ))
        assert payload["ok"], payload["errors"]
        by_id = {row["id"]: row for row in payload["advisories"]}
        assert "ADV-STREAMED-CONTRACT" in by_id
        assert by_id["ADV-STREAMED-CONTRACT"]["severity"] == "INFO"

    def test_lif_draft_has_no_casc_advisory(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={"spiking_family": "lif"}
        ))
        assert payload["ok"], payload["errors"]
        ids = {row["id"] for row in payload["advisories"]}
        assert "ADV-CASC-UNSUPPORTED" not in ids

    def test_erroring_draft_renders_no_hypothetical_advisories(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={
                "spiking_family": "lif",
                "activation_quantization": False,
                "weight_quantization": True,
            }
        ))
        assert not payload["ok"]
        assert payload["advisories"] == []


class TestReviewLaunchLayout:
    """Advisories render as the FIRST Review & Launch card (before Derived
    values); the live rail carries only a compact count badge near the verdict
    pill. String-level pin — the pixel acceptance is the owner's review against
    docs/ux/review_launch_advisories.md."""

    def _html(self):
        return _WIZARD_HTML.read_text()

    def _review_section(self, html):
        start = html.index('data-section-id="review"')
        # Review's cards are divs (class "section"); the next </section> close
        # is therefore the review section's own.
        return html[start:html.index("</section>", start)]

    def test_advisory_block_is_a_review_section_card(self):
        review = self._review_section(self._html())
        assert 'id="advisoryBlock"' in review
        assert 'id="advisoryRail"' in review

    def test_advisory_block_precedes_the_derived_values_card(self):
        review = self._review_section(self._html())
        assert review.index('id="advisoryBlock"') < review.index('data-section="derived"')

    def test_rail_hosts_the_count_badge_next_to_the_verdict_pill(self):
        html = self._html()
        rail = html[html.index('id="liveRail"'):]
        assert 'id="advisoryCountBadge"' in rail
        assert (
            rail.index('id="statusPill"')
            < rail.index('id="advisoryCountBadge"')
            < rail.index('id="assemblyBlock"')
        )

    def test_the_rail_advisory_block_is_gone(self):
        html = self._html()
        assert 'id="advisoryBlock"' not in html[html.index('id="liveRail"'):]
        assert html.count('id="advisoryBlock"') == 1
        assert html.count('id="advisoryRail"') == 1
