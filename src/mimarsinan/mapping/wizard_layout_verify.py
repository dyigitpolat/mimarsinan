"""Compatibility shim — see mimarsinan.mapping.verification.wizard_layout_verify."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.verification.wizard_layout_verify")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
