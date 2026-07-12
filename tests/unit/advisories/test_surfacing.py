"""Surfacing: loud [ADVISORY] prints + machine-readable reporter events, never a gate."""

from mimarsinan.advisories.advisory import SEVERITY_RISK, Advisory
from mimarsinan.advisories.surfacing import ADVISORY_EVENT_KIND, surface_advisories


class RecordingReporter:
    prefix = ""

    def __init__(self):
        self.events = []

    def report(self, metric_name, metric_value, step=None):
        pass

    def console_log(self, metric_name, metric_value):
        pass

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self):
        pass


class LegacyReporter:
    """Predates the event API — surfacing must tolerate it (print stays loud)."""

    prefix = ""

    def report(self, metric_name, metric_value, step=None):
        pass


def _advisory(**over):
    kwargs = dict(
        id="ADV-NOVENA-CHARGE",
        severity=SEVERITY_RISK,
        title="Novena zero-reset breaks charge conservation",
        detail="detail text",
        tentative=True,
        mandate_violation=True,
        suggested_levers=("firing_mode=Default",),
    )
    kwargs.update(over)
    return Advisory(**kwargs)


class TestSurfaceAdvisories:
    def test_prints_loud_advisory_prefix(self, capsys):
        surface_advisories(RecordingReporter(), [_advisory()], context="config")
        out = capsys.readouterr().out
        assert "[ADVISORY][RISK] ADV-NOVENA-CHARGE" in out
        assert "Novena zero-reset breaks charge conservation" in out
        assert "detail text" in out
        assert "firing_mode=Default" in out
        assert "tentative" in out
        assert "mandate-violation" in out

    def test_emits_machine_readable_reporter_events(self):
        reporter = RecordingReporter()
        surface_advisories(reporter, [_advisory()], context="config")
        assert len(reporter.events) == 1
        kind, payload = reporter.events[0]
        assert kind == ADVISORY_EVENT_KIND == "deployment_advisory"
        assert payload["context"] == "config"
        assert payload["id"] == "ADV-NOVENA-CHARGE"
        assert payload["severity"] == "RISK"
        assert payload["mandate_violation"] is True
        assert payload["tentative"] is True
        assert payload["suggested_levers"] == ["firing_mode=Default"]

    def test_silent_on_no_advisories(self, capsys):
        reporter = RecordingReporter()
        surface_advisories(reporter, [], context="config")
        assert reporter.events == []
        assert "[ADVISORY]" not in capsys.readouterr().out

    def test_tolerates_reporters_without_event_api(self, capsys):
        surface_advisories(LegacyReporter(), [_advisory()], context="config")
        assert "[ADVISORY]" in capsys.readouterr().out
