"""Compatibility shim — see mimarsinan.mapping.packing.softcore_mapping."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.packing.softcore_mapping")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
