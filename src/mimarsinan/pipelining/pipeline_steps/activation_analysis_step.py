"""Compatibility shim — aliases implementation module for monkeypatch-safe imports."""

import importlib as _importlib
import sys as _sys

_TARGET = "mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step"
_impl = _importlib.import_module(_TARGET)
_sys.modules[__name__] = _impl
