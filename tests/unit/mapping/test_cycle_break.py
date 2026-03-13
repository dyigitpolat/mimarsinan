"""Regression test: mapping.ir and mapping.soft_core_mapper must be importable in either order without cycles.

After the cycle-break refactor (soft_core_mapper holds SoftCoreMapping; ir imports it only lazily),
both modules must load without ImportError or AttributeError regardless of import order.
"""

import pytest


def test_ir_then_soft_core_mapper_no_cycle():
    """Import ir first, then soft_core_mapper: no circular import or attribute error."""
    import mimarsinan.mapping.ir as ir  # noqa: F401
    import mimarsinan.mapping.soft_core_mapper as soft_core_mapper  # noqa: F401
    assert ir.IRGraph is not None
    assert soft_core_mapper.SoftCoreMapping is not None


def test_soft_core_mapper_then_ir_no_cycle():
    """Import soft_core_mapper first, then ir: no circular import or attribute error."""
    import mimarsinan.mapping.soft_core_mapper as soft_core_mapper  # noqa: F401
    import mimarsinan.mapping.ir as ir  # noqa: F401
    assert soft_core_mapper.SoftCoreMapping is not None
    assert ir.IRGraph is not None
