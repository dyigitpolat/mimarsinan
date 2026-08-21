"""The ODIN core generator: a declared core type + a soma law become RTL."""

from mimarsinan.mapping.export.odin_gen.generate import (
    GeneratedCore,
    generate_core,
    write_generated_core,
)
from mimarsinan.mapping.export.odin_gen.spec import (
    CoreSpec,
    CoreSpecError,
    require_generatable,
)
from mimarsinan.mapping.export.odin_gen.passthrough import (
    StockPassthroughError,
    is_stock_spec,
    stock_core_spec,
)

__all__ = [
    "CoreSpec",
    "CoreSpecError",
    "GeneratedCore",
    "StockPassthroughError",
    "generate_core",
    "is_stock_spec",
    "require_generatable",
    "stock_core_spec",
    "write_generated_core",
]
