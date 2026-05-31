"""Unit tests for the beton cache path logic."""

from __future__ import annotations

import os
import tempfile

import pytest

from mimarsinan.data_handling.ffcv.cache import beton_path_for, cache_root
from mimarsinan.data_handling.ffcv.pipeline_spec import (
    FieldSpec,
    PipelineSpec,
    SplitSpec,
)


def _spec(id_="toy"):
    return PipelineSpec(
        id=id_,
        fields=(FieldSpec(name="image", write_type="RGBImageField"),
                FieldSpec(name="label", write_type="IntField")),
        splits={"train": SplitSpec(shuffle=True),
                "val": SplitSpec(),
                "test": SplitSpec()},
    )


def test_cache_root_respects_env_override(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", tmp)
        assert str(cache_root()).startswith(tmp)


def test_beton_path_includes_spec_hash(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", tmp)
        path = beton_path_for(_spec(), "train")
        parts = path.parts
        assert "toy" in parts
        # one of the parts must be the 12-char hash
        assert any(len(p) == 12 and p.isalnum() for p in parts)
        assert path.name == "train.beton"


def test_distinct_specs_get_distinct_dirs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", tmp)
        spec_a = _spec("a")
        spec_b = _spec("b")
        assert beton_path_for(spec_a, "train").parent != beton_path_for(spec_b, "train").parent


def test_split_synonyms_resolve_to_same_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("MIMARSINAN_FFCV_CACHE_DIR", tmp)
        assert beton_path_for(_spec(), "val") == beton_path_for(_spec(), "validation")
