"""Compatibility shim — see mimarsinan.mapping.platform.mapping_structure."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.platform.mapping_structure")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
