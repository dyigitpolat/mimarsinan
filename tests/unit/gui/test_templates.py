"""Unit tests for mimarsinan.gui.templates."""

import json

import pytest

from mimarsinan.gui.templates import (
    delete_template,
    get_template,
    get_templates_dir,
    list_templates,
    save_template,
)


class TestGetTemplatesDir:
    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", "/custom/templates")
        assert get_templates_dir() == "/custom/templates"

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_TEMPLATES_DIR", raising=False)
        assert get_templates_dir() == "./templates"


class TestListTemplates:
    def test_empty_dir_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        assert list_templates() == []

    def test_with_templates_returns_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        (tmp_path / "template_a.json").write_text(
            json.dumps({"experiment_name": "A", "pipeline_mode": "phased"}),
            encoding="utf-8",
        )
        (tmp_path / "template_b.json").write_text(
            json.dumps({"experiment_name": "B", "pipeline_mode": "vanilla"}),
            encoding="utf-8",
        )
        results = list_templates()
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"template_a", "template_b"}
        for r in results:
            assert "name" in r
            assert "pipeline_mode" in r
            assert "created_at" in r


class TestSaveGetTemplateRoundTrip:
    def test_save_and_get_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        config = {"experiment_name": "Round Trip", "pipeline_mode": "phased", "seed": 123}
        template_id = save_template("Round Trip", config)
        assert template_id == "Round_Trip"
        loaded = get_template(template_id)
        # save_template normalizes to the canonical deployment-config shape,
        # preserving every explicit key verbatim (a declared pipeline_mode
        # included) with experiment_name overridden to the display name.
        assert loaded["experiment_name"] == "Round Trip"
        assert loaded["seed"] == 123
        assert loaded["pipeline_mode"] == "phased"
        assert loaded["deployment_parameters"] == {}
        assert loaded["platform_constraints"] == {}

    def test_save_sanitizes_name_to_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        config = {"experiment_name": "Test"}
        template_id = save_template("My Config v2!", config)
        assert template_id == "My_Config_v2_"
        loaded = get_template(template_id)
        assert loaded["experiment_name"] == "My Config v2!"
        assert config["experiment_name"] == "Test"

    def test_save_template_overwrites_experiment_name_with_template_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        config = {"experiment_name": "old_run_name", "pipeline_mode": "phased", "seed": 1}
        save_template("Display Name", config)
        loaded = get_template("Display_Name")
        assert loaded["experiment_name"] == "Display Name"
        assert loaded["seed"] == 1
        # pipeline_mode is declarable-derived: preserved verbatim on save.
        assert loaded["pipeline_mode"] == "phased"
        assert config["experiment_name"] == "old_run_name"


class TestDeleteTemplate:
    def test_existing_template_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        save_template("to_delete", {"x": 1})
        assert delete_template("to_delete") is True
        assert get_template("to_delete") is None

    def test_missing_template_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        assert delete_template("nonexistent") is False


class TestInvalidTemplateId:
    def test_path_traversal_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", "/some/dir")
        with pytest.raises(ValueError, match="Invalid template id"):
            get_template("../foo")

    def test_slash_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", "/some/dir")
        with pytest.raises(ValueError, match="Invalid template id"):
            get_template("foo/bar")

    def test_delete_invalid_id_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", "/some/dir")
        with pytest.raises(ValueError, match="Invalid template id"):
            delete_template("../evil")


class TestSubdirectoryGroups:
    """Templates one level deep group under their subdirectory name (the tier_0/
    tier_1/tier_2 deployment-mode example groups); the id stays a flat stem so
    the /api routes and ?template_id= links need no slash handling."""

    def test_flat_templates_have_empty_group(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        (tmp_path / "flat.json").write_text(
            json.dumps({"experiment_name": "F", "pipeline_mode": "phased"}),
            encoding="utf-8",
        )
        results = list_templates()
        assert len(results) == 1 and results[0]["group"] == ""

    def test_subdir_templates_carry_their_group(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        sub = tmp_path / "tier_0"
        sub.mkdir()
        (sub / "t0_01.json").write_text(
            json.dumps({"experiment_name": "M", "pipeline_mode": "phased"}), encoding="utf-8")
        (tmp_path / "flat.json").write_text(
            json.dumps({"experiment_name": "F", "pipeline_mode": "vanilla"}), encoding="utf-8")
        groups = {r["id"]: r["group"] for r in list_templates()}
        assert groups == {"t0_01": "tier_0", "flat": ""}

    def test_get_template_resolves_a_subdir_template(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        sub = tmp_path / "tier_1"
        sub.mkdir()
        (sub / "t1_01.json").write_text(json.dumps({"experiment_name": "M"}), encoding="utf-8")
        assert get_template("t1_01")["experiment_name"] == "M"

    def test_delete_template_resolves_a_subdir_template(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        sub = tmp_path / "tier_2"
        sub.mkdir()
        p = sub / "t2_01.json"
        p.write_text(json.dumps({"experiment_name": "M"}), encoding="utf-8")
        assert delete_template("t2_01") is True and not p.exists()

    def test_only_recurses_one_level(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        deep = tmp_path / "tier_0" / "nested"
        deep.mkdir(parents=True)
        (deep / "x.json").write_text(json.dumps({"experiment_name": "X"}), encoding="utf-8")
        assert list_templates() == []

    def test_flat_template_shadows_a_subdir_stem(self, tmp_path, monkeypatch):
        # A flat user template wins over a same-stem subdir example (searched first).
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        (tmp_path / "dup.json").write_text(json.dumps({"experiment_name": "FLAT"}), encoding="utf-8")
        sub = tmp_path / "tier_0"
        sub.mkdir()
        (sub / "dup.json").write_text(json.dumps({"experiment_name": "SUB"}), encoding="utf-8")
        assert get_template("dup")["experiment_name"] == "FLAT"


class TestManifestExcluded:
    def test_manifest_json_is_not_a_template(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        sub = tmp_path / "tier_0"
        sub.mkdir()
        (sub / "manifest.json").write_text(json.dumps({"tier": 0, "runs": []}), encoding="utf-8")
        (sub / "t0_01.json").write_text(json.dumps({"experiment_name": "M"}), encoding="utf-8")
        ids = {r["id"] for r in list_templates()}
        assert ids == {"t0_01"}
