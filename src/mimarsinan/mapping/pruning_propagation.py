"""Compatibility shim — see mimarsinan.mapping.pruning.pruning_propagation."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.pruning.pruning_propagation")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
