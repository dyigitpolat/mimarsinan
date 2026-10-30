"""[ODIN H0] the card power sampler: one parser, two routes, no silent zero.

``scripts/hacc/package/host/odin_power_sampler.py`` reads the card's electrical
and thermal reports, by ``xrt-smi examine`` (route A) or pyxrt ``get_info``
(route B). Its parser must survive the schema moving under it (wrappers,
renamings, numbers for strings, missing tables) and must REFUSE, by type, a
report that carries no board power reading. The schema fixture next to this
file is constructed from the XRT 2024.2 sources, not recorded on a card; its
provenance says so, and the H1 pilot's first raw reports replace it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "scripts" / "hacc" / "package" / "host"
SAMPLER_PATH = HOST / "odin_power_sampler.py"
FAKE_PATH = HOST / "fake_pyxrt_for_selftest.py"
SCHEMA = Path(__file__).with_name("xrt_2024_2_electrical_thermal_schema.json")
BDF = "0000:81:00.1"


@pytest.fixture(scope="module")
def sampler():
    spec = importlib.util.spec_from_file_location("packaged_odin_power_sampler", SAMPLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def _copy(document):
    return json.loads(json.dumps(document))


class TestTheFixtureSaysWhatItIs:
    def test_it_is_labelled_constructed_not_recorded(self, schema):
        provenance = schema["provenance"]
        assert provenance["class"].startswith("CONSTRUCTED, NOT RECORDED")
        assert provenance["branch"] == "2024.2"
        assert any("sensor.cpp" in source for source in provenance["sources"])


class TestTheParserReadsBothRoutes:
    def test_the_examine_shape(self, sampler, schema):
        reading = sampler.parse_report(json.dumps(schema["xrt_smi_examine"]), bdf=BDF)
        assert reading["power_w"] == pytest.approx(22.413207)
        assert reading["power_field"] == "electrical.power_consumption_watts"
        assert reading["power_max_w"] == 150.0
        assert reading["bdf"] == BDF
        assert reading["temps_c"] == {"pcb_top_front": 38.0, "fpga0": 44.0,
                                      "fpga_hbm": 41.0}
        assert reading["rails"]["12v_pex"]["watts"] == pytest.approx(12.143 * 1.420)
        assert reading["rails"]["12v_aux"]["amps"] is None  # is_present false

    def test_the_get_info_shape(self, sampler, schema):
        document = {"device_id": BDF, "electrical": schema["pyxrt_get_info_electrical"],
                    "thermal": schema["pyxrt_get_info_thermal"]}
        reading = sampler.parse_report(document, bdf=BDF)
        assert reading["power_w"] == pytest.approx(22.413207)
        assert reading["temps_c"] == {"fpga0": 44.0}

    def test_the_bare_electrical_tree(self, sampler, schema):
        reading = sampler.parse_report(schema["pyxrt_get_info_electrical"])
        assert reading["power_w"] == pytest.approx(22.413207)
        assert reading["temps_c"] == {}


class TestTheParserToleratesSchemaChange:
    @pytest.mark.parametrize("mutate, expected", [
        (lambda e: e.update(power_consumption_watts=21.5), 21.5),
        (lambda e: e.update(power_consumption_watts="21.5 W"), 21.5),
        (lambda e: e.update(POWER_CONSUMPTION_WATTS=e.pop("power_consumption_watts")),
         22.413207),
        (lambda e: e.update(power_watts=e.pop("power_consumption_watts")), 22.413207),
        (lambda e: e.update(board_power_watts=e.pop("power_consumption_watts")), 22.413207),
        (lambda e: e.pop("power_rails"), 22.413207),
        (lambda e: e.update(new_field_from_a_later_xrt={"anything": [1, 2]}), 22.413207),
    ], ids=["number", "unit-suffix", "upper-case", "renamed-power_watts",
            "renamed-board_power_watts", "no-rails", "extra-field"])
    def test_a_changed_electrical_tree_still_parses(self, sampler, schema, mutate, expected):
        document = _copy(schema["xrt_smi_examine"])
        mutate(document["devices"][0]["electrical"])
        assert sampler.parse_report(document, bdf=BDF)["power_w"] == pytest.approx(expected)

    def test_an_extra_wrapper_level_parses(self, sampler, schema):
        wrapped = {"result": {"report": schema["xrt_smi_examine"]}}
        assert sampler.parse_report(wrapped, bdf=BDF)["power_w"] == pytest.approx(22.413207)

    def test_a_numeric_temperature_parses(self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        document["devices"][0]["thermals"][1]["temp_C"] = 47
        assert sampler.parse_report(document, bdf=BDF)["temps_c"]["fpga0"] == 47.0

    def test_the_renamed_key_is_recorded(self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        electrical = document["devices"][0]["electrical"]
        electrical["power_watts"] = electrical.pop("power_consumption_watts")
        assert sampler.parse_report(document)["power_field"] == "electrical.power_watts"

    def test_the_signature_follows_keys_not_values(self, sampler, schema):
        base = schema["xrt_smi_examine"]["devices"][0]
        same_keys = _copy(base)
        same_keys["electrical"]["power_consumption_watts"] = "30.0"
        renamed = _copy(base)
        renamed["electrical"]["power_watts"] = renamed["electrical"].pop(
            "power_consumption_watts")
        assert sampler.schema_signature(base) == sampler.schema_signature(same_keys)
        assert sampler.schema_signature(base) != sampler.schema_signature(renamed)


class TestTheParserRefusesAMissingReading:
    @pytest.mark.parametrize("mutate, needle", [
        (lambda e: e.pop("power_consumption_watts"), "no board power field"),
        (lambda e: e.update(power_consumption_watts="N/A"), "not a number"),
        (lambda e: e.update(power_consumption_watts=""), "not a number"),
        (lambda e: e.update(power_consumption_watts=None), "not a number"),
        (lambda e: e.update(power_consumption_watts="12 mW"), "not a number"),
        (lambda e: e.update(power_consumption_watts="0.000000"), "outside"),
        (lambda e: e.update(power_consumption_watts="-3"), "outside"),
        (lambda e: e.update(power_consumption_watts="9999"), "outside"),
    ], ids=["absent", "N/A", "empty", "null", "milliwatts", "zero", "negative",
            "implausible"])
    def test_a_report_without_a_reading_refuses(self, sampler, schema, mutate, needle):
        document = _copy(schema["xrt_smi_examine"])
        mutate(document["devices"][0]["electrical"])
        with pytest.raises(sampler.OdinTelemetryMissing, match=needle):
            sampler.parse_report(document, bdf=BDF)

    def test_the_power_limit_is_never_read_as_the_reading(self, sampler):
        with pytest.raises(sampler.OdinTelemetryMissing, match="no board power field"):
            sampler.parse_report({"electrical": {"power_consumption_max_watts": "150"}})

    def test_no_sensors_present_refuses_and_says_so(self, sampler):
        with pytest.raises(sampler.OdinTelemetryMissing, match="No sensors present"):
            sampler.parse_report({"electrical": {"msg": "No sensors present"}})

    def test_text_that_is_not_json_refuses(self, sampler):
        with pytest.raises(sampler.OdinTelemetryMissing, match="not JSON"):
            sampler.parse_report("ERROR: device is busy")

    def test_a_report_about_the_other_card_refuses(self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        document["devices"][0]["device_id"] = "0000:c1:00.1"
        with pytest.raises(sampler.OdinTelemetryMismatch, match="another card"):
            sampler.parse_report(document, bdf=BDF)

    def test_two_cards_and_no_pin_refuses(self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        other = _copy(document["devices"][0])
        other["device_id"] = "0000:c1:00.1"
        document["devices"].append(other)
        with pytest.raises(sampler.OdinTelemetryMismatch, match="no BDF pins one"):
            sampler.parse_report(document)
        assert sampler.parse_report(document, bdf="0000:c1:00.1")["bdf"] == "0000:c1:00.1"

    def test_a_bdf_without_a_domain_is_the_same_card(self, sampler, schema):
        assert sampler.parse_report(schema["xrt_smi_examine"], bdf="81:00.1")["bdf"] == BDF


def _stub_smi(tmp_path: Path, schema, *, exit_code: int = 0, to_stdout: bool = False) -> Path:
    """An executable standing in for xrt-smi: writes the fixture where -o says."""
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(schema["xrt_smi_examine"]), encoding="utf-8")
    argv_log = tmp_path / "argv.json"
    script = tmp_path / "bin" / "xrt-smi"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"""#!{sys.executable}
import json, sys
args = sys.argv[1:]
open({str(argv_log)!r}, "w").write(json.dumps(args))
if {exit_code}:
    sys.stderr.write("ERROR: card is busy\\n")
    sys.exit({exit_code})
text = open({str(payload)!r}).read()
if {to_stdout!r}:
    print("banner line"); print(text)
else:
    open(args[args.index("-o") + 1], "w").write(text)
""", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class TestRouteA:
    def test_one_examine_per_reading_pinned_to_the_bdf(self, sampler, schema, tmp_path):
        smi = _stub_smi(tmp_path, schema)
        source = sampler.SmiSource(bdf=BDF, scratch=str(tmp_path / "scratch"), binary=str(smi))
        reading = sampler.parse_report(source.read(), bdf=BDF)
        assert reading["power_w"] == pytest.approx(22.413207)
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[:3] == ["examine", "-d", BDF]
        assert argv[argv.index("-r") + 1:argv.index("-r") + 3] == ["electrical", "thermal"]
        assert "JSON" in argv and "--force" in argv

    def test_the_command_can_be_respelled_without_editing_code(self, sampler, schema,
                                                              tmp_path, monkeypatch):
        smi = _stub_smi(tmp_path, schema)
        monkeypatch.setenv(sampler.SMI_EXAMINE_ENV, json.dumps(
            ["examine", "--device", "{bdf}", "--report", "electrical", "--format",
             "JSON", "--output", "{out}", "--force", "-o", "{out}"]))
        source = sampler.SmiSource(bdf=BDF, scratch=str(tmp_path / "s"), binary=str(smi))
        source.read()
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[:3] == ["examine", "--device", BDF]

    def test_json_printed_instead_of_written_is_taken(self, sampler, schema, tmp_path):
        smi = _stub_smi(tmp_path, schema, to_stdout=True)
        source = sampler.SmiSource(bdf=BDF, scratch=str(tmp_path / "s"), binary=str(smi))
        assert sampler.parse_report(source.read(), bdf=BDF)["power_w"] > 0

    def test_a_failing_tool_is_a_source_error_not_a_reading(self, sampler, schema, tmp_path):
        smi = _stub_smi(tmp_path, schema, exit_code=3)
        source = sampler.SmiSource(bdf=BDF, scratch=str(tmp_path / "s"), binary=str(smi))
        with pytest.raises(sampler.OdinTelemetrySourceError, match="card is busy"):
            source.read()

    def test_xrt_smi_is_preferred_over_xbutil(self, sampler, tmp_path, monkeypatch):
        root = tmp_path / "xrt"
        for name in ("xbutil", "xrt-smi"):
            path = root / "bin" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)
        monkeypatch.setenv("XILINX_XRT", str(root))
        assert sampler.resolve_smi().endswith("xrt-smi")

    def test_route_a_needs_the_bdf(self, sampler, tmp_path):
        with pytest.raises(sampler.OdinTelemetrySourceError, match="BDF"):
            sampler.SmiSource(bdf="", scratch=str(tmp_path))


class TestRouteB:
    def test_get_info_through_the_fake(self, sampler):
        source = sampler.PyxrtSource(fake_pyxrt=str(FAKE_PATH))
        assert source.bdf == BDF
        reading = sampler.parse_report(source.read(), bdf=BDF)
        assert reading["power_w"] > 0 and "fpga0" in reading["temps_c"]

    def test_a_pinned_bdf_that_is_not_the_device_refuses(self, sampler):
        with pytest.raises(sampler.OdinTelemetryMismatch):
            sampler.PyxrtSource(fake_pyxrt=str(FAKE_PATH), bdf="0000:c1:00.1")


class _Script:
    name = "scripted"

    def __init__(self, documents):
        self.documents = list(documents)

    def describe(self):
        return {"source": self.name}

    def read(self):
        return self.documents.pop(0) if self.documents else self.final

    final = '{"electrical": {"power_consumption_watts": "20.0"}}'

    def close(self):
        pass


class TestTheSamplerLoop:
    def test_every_read_is_a_line_and_a_refusal_is_recorded_not_zeroed(
            self, sampler, schema, tmp_path):
        documents = [json.dumps(schema["xrt_smi_examine"]),
                     '{"electrical": {"power_consumption_watts": "N/A"}}',
                     json.dumps(schema["xrt_smi_examine"])]
        out = tmp_path / "samples.jsonl"
        count = sampler.sample(_Script(documents), out_path=str(out), bdf=BDF,
                               interval_s=0.0, max_samples=4)
        headers, rows = sampler.load_samples(str(out))
        assert count == 4 and len(rows) == 4 and headers[0]["clock"] == "time.monotonic"
        assert [row["ok"] for row in rows] == [True, False, True, True]
        assert rows[1]["refusal"] == "OdinTelemetryMissing" and "power_w" not in rows[1]
        assert all(rows[i]["t_mono"] < rows[i + 1]["t_mono"] for i in range(3))
        kept = sorted(os.listdir(tmp_path / "raw"))
        # One raw report per distinct key structure, not one per read: the N/A
        # report and the last one share theirs.
        assert len(kept) == 2
        assert rows[1]["signature"] == rows[3]["signature"] != rows[0]["signature"]

    def test_a_stop_file_ends_the_loop(self, sampler, tmp_path):
        stop = tmp_path / "stop"
        stop.write_text("stop")
        out = tmp_path / "samples.jsonl"
        assert sampler.sample(_Script([]), out_path=str(out), bdf=None, interval_s=0.0,
                              stop_file=str(stop)) == 1

    def test_a_torn_last_line_is_skipped(self, sampler, tmp_path):
        out = tmp_path / "samples.jsonl"
        out.write_text('{"header": true}\n{"ok": true, "t_mono": 1.0}\n{"ok": tr')
        headers, rows = sampler.load_samples(str(out))
        assert len(headers) == 1 and len(rows) == 1


class TestTheCommandLine:
    def test_parse_exits_three_on_a_refused_report(self, sampler, schema, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps(schema["xrt_smi_examine"]))
        bad = tmp_path / "bad.json"
        bad.write_text('{"electrical": {"msg": "No sensors present"}}')
        assert sampler.main(["parse", "--bdf", BDF, str(good)]) == 0
        assert sampler.main(["parse", str(good), str(bad)]) == 3

    def test_probe_chooses_a_route_and_keeps_its_raw_report(self, sampler, tmp_path):
        out = tmp_path / "probe"
        code = sampler.main(["probe", "--routes", "pyxrt", "--reads", "2",
                             "--fake-pyxrt", str(FAKE_PATH), "--out", str(out)])
        report = json.loads((out / "probe.json").read_text())
        assert code == 0 and report["chosen"] == "pyxrt"
        assert (out / "raw_pyxrt.json").is_file()


class TestTheE64Reading:
    """E64's acceptance as this sampler reads it: the board power reading is
    required; rails and thermals are optional, recorded absent, never zero; the
    recorded fixture exists only after the pilot (run_plan.md)."""

    def test_a_current_the_card_reports_absent_is_recorded_absent_not_zero(
            self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        aux = next(rail for rail in document["devices"][0]["electrical"]["power_rails"]
                   if rail["id"] == "12v_aux")
        assert aux["current"] == {"amps": "0.000", "is_present": "false"}
        rail = sampler.parse_report(document, bdf=BDF)["rails"]["12v_aux"]
        assert rail["volts"] == pytest.approx(12.15)
        assert rail["amps"] is None and rail["watts"] is None

    def test_a_thermal_the_card_reports_absent_is_left_out_not_zero(self, sampler, schema):
        document = _copy(schema["xrt_smi_examine"])
        document["devices"][0]["thermals"].append(
            {"location_id": "cage_temp_9", "temp_C": "0", "is_present": "false"})
        assert "cage_temp_9" not in sampler.parse_report(document, bdf=BDF)["temps_c"]

    def test_no_rail_table_and_no_thermal_table_still_give_a_reading(self, sampler):
        reading = sampler.parse_report({"electrical": {"power_consumption_watts": "19.0"}})
        assert reading["power_w"] == 19.0
        assert reading["rails"] == {} and reading["temps_c"] == {}

    def test_rails_alone_are_not_a_board_power_reading(self, sampler, schema):
        electrical = _copy(schema["xrt_smi_examine"]["devices"][0]["electrical"])
        electrical.pop("power_consumption_watts")
        with pytest.raises(sampler.OdinTelemetryMissing, match="no board power field"):
            sampler.parse_report({"electrical": electrical})

    def test_the_module_states_the_reading(self, sampler):
        text = " ".join(sampler.__doc__.split())
        assert "the board power reading is required" in text
        assert "rails and thermals are optional, recorded absent, never zero" in text
        assert "recorded fixture" in text and "after the pilot" in text


class TestItStaysUploadable:
    def test_standard_library_only(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        for needle in ("import numpy", "import torch", "from mimarsinan",
                       "import mimarsinan", "\nimport pyxrt"):
            assert needle not in source, needle
