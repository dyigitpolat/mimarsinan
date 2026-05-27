"""Compatibility shim — see mimarsinan.mapping.packing.hybrid_hardcore_mapping."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("mimarsinan.mapping.packing.hybrid_hardcore_mapping")
_sys.modules[__name__] = _impl
