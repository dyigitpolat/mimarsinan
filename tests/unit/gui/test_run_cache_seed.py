"""Tests for seeding a new run directory with pipeline cache from a previous run."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mimarsinan.gui.run_cache_seed import (
    copy_pipeline_cache_from_previous_run,
    copy_steps_json_from_previous_run,
)


@pytest.fixture
def tmp_gen_root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_copy_pipeline_cache_basic(tmp_gen_root: Path) -> None:
    prev = tmp_gen_root / "myexp_phased_deployment_run_20250101_120000"
    prev.mkdir(parents=True)
    meta = {"Model Building.model": ["basic", "Model Building.model"]}
    (prev / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (prev / "Model Building.model.json").write_text('{"x": 1}', encoding="utf-8")

    dest = tmp_gen_root / "new_run"
    dest.mkdir()

    copy_pipeline_cache_from_previous_run(str(tmp_gen_root), prev.name, str(dest))

    assert (dest / "metadata.json").is_file()
    assert json.loads((dest / "metadata.json").read_text(encoding="utf-8")) == meta
    assert json.loads((dest / "Model Building.model.json").read_text(encoding="utf-8")) == {"x": 1}


def test_invalid_run_id_skips(tmp_gen_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    dest = tmp_gen_root / "d"
    dest.mkdir()
    copy_pipeline_cache_from_previous_run(str(tmp_gen_root), "../../etc/passwd", str(dest))
    assert not (dest / "metadata.json").exists()


def test_no_metadata_skips(tmp_gen_root: Path) -> None:
    prev = tmp_gen_root / "empty_run"
    prev.mkdir()
    dest = tmp_gen_root / "dest"
    dest.mkdir()
    copy_pipeline_cache_from_previous_run(str(tmp_gen_root), prev.name, str(dest))
    assert not (dest / "metadata.json").exists()


def test_copy_steps_json_from_previous_run(tmp_gen_root: Path) -> None:
    prev = tmp_gen_root / "run_a"
    gui = prev / "_GUI_STATE"
    gui.mkdir(parents=True)
    (gui / "steps.json").write_text('{"steps": {"S1": {"end_time": 1.0}}}', encoding="utf-8")
    dest = tmp_gen_root / "run_b"
    dest.mkdir()
    copy_steps_json_from_previous_run(str(tmp_gen_root), prev.name, str(dest))
    assert (dest / "_GUI_STATE" / "steps.json").is_file()
    import json

    data = json.loads((dest / "_GUI_STATE" / "steps.json").read_text(encoding="utf-8"))
    assert data["steps"]["S1"]["end_time"] == 1.0
