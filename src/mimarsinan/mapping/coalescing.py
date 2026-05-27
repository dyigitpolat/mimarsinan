"""Compatibility shim — see mimarsinan.mapping.platform.coalescing."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.platform.coalescing")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
