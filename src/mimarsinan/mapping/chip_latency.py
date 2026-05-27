"""Compatibility shim — see mimarsinan.mapping.latency.chip."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("mimarsinan.mapping.latency.chip")
_sys.modules[__name__] = _impl
