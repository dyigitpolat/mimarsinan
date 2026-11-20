"""A sealed path refuses every open by name for the rest of the process (held-out data stay closed until their read)."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.data_handling import sealed_paths as sp
from mimarsinan.data_handling.sealed_paths import (
    SealDeclarationRefusal, SealedPathRefusal, seal_paths, sealed_by, sealed_patterns,
)

_SRC = str(Path(sp.__file__).resolve().parents[2])


def _file(root: Path, rel: str, data: bytes = b"held out") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _opens(path: Path):
    """Every reader the toolchain's data path uses, each opening ``path`` by name."""
    return {
        "open rb": lambda: open(path, "rb"),
        "open wb": lambda: open(path, "wb"),
        "os.open": lambda: os.open(path, os.O_RDONLY),
        "numpy.load": lambda: np.load(path),
        "torch.load": lambda: torch.load(path),
        "tarfile.open": lambda: tarfile.open(path),
        "pickle via open": lambda: pickle.load(open(path, "rb")),
    }


class TestSealedFile:
    def test_every_reader_of_a_sealed_file_is_refused_by_name(self, tmp_path):
        held = _file(tmp_path, "batches/test")
        seal_paths([str(held)])
        for reader, call in _opens(held).items():
            with pytest.raises(SealedPathRefusal) as caught:
                call()
            assert str(held) in str(caught.value), reader
            assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, PermissionError)

    def test_a_sibling_and_a_prefix_sibling_still_open(self, tmp_path):
        held = _file(tmp_path, "batches/test")
        sibling = _file(tmp_path, "batches/train")
        prefix = _file(tmp_path, "batches/test2")
        seal_paths([str(held)])
        assert sibling.read_bytes() == b"held out"
        assert prefix.read_bytes() == b"held out"
        assert sealed_by(sibling) is None and sealed_by(prefix) is None
        assert sealed_by(held) == str(held.resolve())

    def test_a_relative_seal_resolves_against_the_working_directory(self, tmp_path, monkeypatch):
        held = _file(tmp_path, "rel/test")
        monkeypatch.chdir(tmp_path)
        added = seal_paths(["rel/test"])
        assert added == (str(held.resolve()),)
        with pytest.raises(SealedPathRefusal):
            open("rel/test", "rb")

    def test_a_link_to_a_sealed_file_is_refused(self, tmp_path):
        held = _file(tmp_path, "real/test")
        link = tmp_path / "link_to_test"
        link.symlink_to(held)
        seal_paths([str(held)])
        with pytest.raises(SealedPathRefusal):
            open(link, "rb")

    def test_a_missing_file_is_sealed_before_it_exists(self, tmp_path):
        held = tmp_path / "later" / "test"
        seal_paths([str(held)])
        held.parent.mkdir()
        with pytest.raises(SealedPathRefusal):
            held.write_bytes(b"extracted")
        assert not held.exists()


class TestSealedDirectoryAndPattern:
    def test_a_sealed_directory_refuses_every_file_under_it(self, tmp_path):
        inner = _file(tmp_path, "archive/inner/data.bin")
        seal_paths([str(tmp_path / "archive")])
        with pytest.raises(SealedPathRefusal):
            open(inner, "rb")
        assert sorted(os.listdir(tmp_path / "archive")) == ["inner"]  # listing is not reading

    def test_a_pattern_seals_the_matching_name_under_any_root(self, tmp_path):
        a = _file(tmp_path, "one/cifar-like/test")
        b = _file(tmp_path, "two/deeper/cifar-like/test")
        train = _file(tmp_path, "one/cifar-like/train")
        pattern = f"*/{tmp_path.name}/*/cifar-like/test"
        seal_paths([pattern])
        for held in (a, b):
            with pytest.raises(SealedPathRefusal):
                open(held, "rb")
        assert train.read_bytes() == b"held out"
        assert sealed_by(a) == pattern

    def test_a_link_named_like_a_pattern_is_refused_even_when_its_target_is_not(self, tmp_path):
        real = _file(tmp_path, "plain/data.bin")
        named = tmp_path / "aliased" / "cifar-alias"
        named.parent.mkdir()
        named.symlink_to(real.parent)
        seal_paths([f"*/{tmp_path.name}/aliased/cifar-alias/*"])
        with pytest.raises(SealedPathRefusal):
            open(named / "data.bin", "rb")
        assert real.read_bytes() == b"held out"  # the target itself is not sealed

    def test_a_relative_pattern_is_refused_by_name(self):
        with pytest.raises(SealDeclarationRefusal, match="pattern") as caught:
            seal_paths(["batches/*/test"])
        assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, ValueError)


class TestSealLifetime:
    def test_seals_only_tighten(self, tmp_path):
        first = _file(tmp_path, "a/test")
        second = _file(tmp_path, "b/test")
        seal_paths([str(first)])
        seal_paths([str(second)])
        active = sealed_patterns()
        assert str(first.resolve()) in active and str(second.resolve()) in active
        assert not any(hasattr(sp, name) for name in ("unseal", "clear", "reset"))

    def test_no_seal_installs_no_hook_and_an_empty_seal_installs_none(self, tmp_path):
        probe = _file(tmp_path, "free/test")
        code = (
            "import json, sys\n"
            "from mimarsinan.data_handling import sealed_paths as sp\n"
            "before = sp.hook_installed()\n"
            "sp.seal_paths([])\n"
            "empty = sp.hook_installed()\n"
            f"data = open({str(probe)!r}, 'rb').read()\n"
            f"sp.seal_paths([{str(probe)!r}])\n"
            "after = sp.hook_installed()\n"
            "try:\n"
            f"    open({str(probe)!r}, 'rb')\n"
            "    refused = False\n"
            "except sp.SealedPathRefusal:\n"
            "    refused = True\n"
            "print(json.dumps([before, empty, data.decode(), after, refused]))\n"
        )
        env = {**os.environ, "PYTHONPATH": _SRC}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        assert json.loads(out.stdout.strip().splitlines()[-1]) == [False, False, "held out", True, True]


class TestThirdPartyReaders:
    def test_the_toolchain_cifar100_provider_cannot_open_a_sealed_dataset_folder(self, tmp_path):
        from mimarsinan.data_handling.data_providers.cifar100_data_provider import CIFAR100_DataProvider

        _file(tmp_path, "cifar-100-python/train", b"not the batch")
        _file(tmp_path, "cifar-100-python/test", b"not the batch")
        seal_paths([str(tmp_path / "cifar-100-python")])
        with pytest.raises(SealedPathRefusal):
            CIFAR100_DataProvider(str(tmp_path))
