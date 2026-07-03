"""_sanafe version discovery: missing pip metadata falls back; other errors propagate."""

import importlib.metadata as md
import sys
import types

import pytest

import mimarsinan.chip_simulation.sanafe.arch_synth.spec as spec


def _install_fake_sanafe(monkeypatch, version_attr=None):
    fake = types.ModuleType("sanafe")
    if version_attr is not None:
        fake.__version__ = version_attr
    monkeypatch.setitem(sys.modules, "sanafe", fake)
    monkeypatch.setattr(spec, "_SANAFE_MODULE", None)
    return fake


def test_metadata_version_used(monkeypatch):
    fake = _install_fake_sanafe(monkeypatch)
    monkeypatch.setattr(md, "version", lambda name: "2.1.1")
    assert spec._sanafe() is fake


def test_package_not_found_falls_back_to_module_version(monkeypatch):
    fake = _install_fake_sanafe(monkeypatch, version_attr="2.1.1")

    def not_found(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", not_found)
    assert spec._sanafe() is fake


def test_unsupported_version_still_blocked_after_fallback(monkeypatch):
    _install_fake_sanafe(monkeypatch, version_attr="2.2.6")

    def not_found(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", not_found)
    with pytest.raises(RuntimeError, match="unsupported"):
        spec._sanafe()


def test_unexpected_metadata_error_propagates(monkeypatch):
    _install_fake_sanafe(monkeypatch)

    def boom(name):
        raise RuntimeError("metadata backend broken")

    monkeypatch.setattr(md, "version", boom)
    with pytest.raises(RuntimeError, match="metadata backend broken"):
        spec._sanafe()
