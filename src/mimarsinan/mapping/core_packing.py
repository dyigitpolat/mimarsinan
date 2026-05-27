"""Compatibility shim — see mimarsinan.mapping.packing.core_packing."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("mimarsinan.mapping.packing.core_packing")
_sys.modules[__name__] = _impl
