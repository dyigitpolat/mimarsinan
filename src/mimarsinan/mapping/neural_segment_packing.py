"""Compatibility shim — see mimarsinan.mapping.packing.neural_segment_packing."""

import importlib as _importlib

_mod = _importlib.import_module("mimarsinan.mapping.packing.neural_segment_packing")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
