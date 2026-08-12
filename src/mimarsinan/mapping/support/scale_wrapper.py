"""Peel a ``ScaleNormalizingWrapper`` to reach the op module it wraps.

Its own module so a measurement helper can unwrap without importing the gates
that report the measurement.
"""

from __future__ import annotations


def is_scale_wrapper(module) -> bool:
    return type(module).__name__ == "ScaleNormalizingWrapper" and hasattr(
        module, "module"
    )


def unwrap_scale_wrapper(module):
    """Peel a ``ScaleNormalizingWrapper`` to reach the wrapped op module."""
    while is_scale_wrapper(module):
        module = module.module
    return module
