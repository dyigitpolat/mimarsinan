"""Compatibility shim — see mimarsinan.mapping.export.chip_export."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.export.chip_export")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
