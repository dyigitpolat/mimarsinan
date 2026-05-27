"""Compatibility shim — see mimarsinan.mapping.pruning.ir_pruning."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.pruning.ir_pruning")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
