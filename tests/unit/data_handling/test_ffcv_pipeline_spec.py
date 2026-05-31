"""Unit tests for the FFCV pipeline spec — pure logic, no FFCV runtime needed."""

from __future__ import annotations

import pytest

from mimarsinan.data_handling.ffcv.pipeline_spec import (
    FieldSpec,
    PipelineSpec,
    SplitSpec,
    normalize_split_name,
)


def _toy_spec() -> PipelineSpec:
    return PipelineSpec(
        id="toy",
        fields=(
            FieldSpec(name="image", write_type="RGBImageField"),
            FieldSpec(name="label", write_type="IntField"),
        ),
        splits={
            "train": SplitSpec(transforms=(("image", "ToTensor", {}),), shuffle=True, drop_last=True),
            "val":   SplitSpec(transforms=(("image", "ToTensor", {}),), shuffle=False),
        },
    )


class TestNormalizeSplitName:
    def test_train_synonyms(self):
        assert normalize_split_name("train") == "train"
        assert normalize_split_name("Training") == "train"

    def test_val_synonyms(self):
        assert normalize_split_name("val") == "val"
        assert normalize_split_name("validation") == "val"
        assert normalize_split_name("VAL") == "val"

    def test_test_synonyms(self):
        assert normalize_split_name("test") == "test"
        assert normalize_split_name("Testing") == "test"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_split_name("predict")


class TestPipelineSpecHash:
    def test_hash_is_stable_across_constructions(self):
        a = _toy_spec()
        b = _toy_spec()
        assert a.stable_hash() == b.stable_hash()

    def test_hash_changes_on_field_change(self):
        a = _toy_spec()
        b = PipelineSpec(
            id=a.id,
            fields=a.fields + (FieldSpec(name="weight", write_type="FloatField"),),
            splits=a.splits,
        )
        assert a.stable_hash() != b.stable_hash()

    def test_hash_changes_on_image_field_write_kwargs_change(self):
        """Changing ``max_resolution`` (which the FFCV layer derives from
        ``_preprocessing_spec.resize_to``) must invalidate the cache."""
        a = _toy_spec()
        b = PipelineSpec(
            id=a.id,
            fields=tuple(
                FieldSpec(name=f.name, write_type=f.write_type,
                          write_kwargs={"max_resolution": 224})
                if f.name == "image" else f
                for f in a.fields
            ),
            splits=a.splits,
        )
        assert a.stable_hash() != b.stable_hash()

    def test_hash_is_short(self):
        h = _toy_spec().stable_hash()
        assert isinstance(h, str)
        assert 8 <= len(h) <= 16
