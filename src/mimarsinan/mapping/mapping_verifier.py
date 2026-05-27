"""Compatibility shim — see mimarsinan.mapping.verification.mapping_verifier."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.verification.mapping_verifier")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
