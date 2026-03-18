"""Unit tests for mimarsinan.gui.templates."""

import json

import pytest

from mimarsinan.gui.templates import (
    get_templates_dir,
    list_templates,
    save_template,
    get_template,
    delete_template,
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
        assert loaded == config

    def test_save_sanitizes_name_to_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
        config = {"experiment_name": "Test"}
        template_id = save_template("My Config v2!", config)
        assert template_id == "My_Config_v2_"
        loaded = get_template(template_id)
        assert loaded == config


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
