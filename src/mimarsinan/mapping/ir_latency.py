"""Compatibility shim — see mimarsinan.mapping.latency.ir."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("mimarsinan.mapping.latency.ir")
_sys.modules[__name__] = _impl
