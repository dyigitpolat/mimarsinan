"""Tests for common file utilities."""

import pytest
import os
import numpy as np

from mimarsinan.common.file_utils import (
    prepare_containing_directory,
    input_to_file,
)


class TestPrepareContainingDirectory:
    def test_creates_parent(self, tmp_path):
        target = str(tmp_path / "sub" / "dir" / "file.txt")
        prepare_containing_directory(target)
        assert os.path.isdir(str(tmp_path / "sub" / "dir"))

    def test_existing_directory_is_noop(self, tmp_path):
        target = str(tmp_path / "file.txt")
        prepare_containing_directory(target)


class TestInputToFile:
    def test_writes_file(self, tmp_path):
        filepath = str(tmp_path / "input.txt")
        data = np.array([1.0, 2.0, 3.0])
        input_to_file(data, 5, filepath)
        assert os.path.isfile(filepath)

        with open(filepath) as f:
            content = f.read().strip()
        parts = content.split()
        assert parts[0] == "5"
        assert parts[1] == "1"
        assert parts[2] == "3"
        assert float(parts[3]) == pytest.approx(1.0)

    def test_different_targets(self, tmp_path):
        for target in [0, 3, 9]:
            filepath = str(tmp_path / f"inp_{target}.txt")
            data = np.array([0.5])
            input_to_file(data, target, filepath)
            with open(filepath) as f:
                assert f.read().strip().startswith(str(target))
