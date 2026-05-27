"""Compatibility shim — see mimarsinan.mapping.verification.layout_request."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.verification.layout_request")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
