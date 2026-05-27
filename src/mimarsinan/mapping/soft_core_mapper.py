"""Compatibility shim — see mimarsinan.mapping.packing.soft_core_mapper."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.packing.soft_core_mapper")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
