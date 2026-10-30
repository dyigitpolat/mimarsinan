"""The variant FABRIC: one ``CoreSpec`` becomes the core AND the kernel around it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from mimarsinan.mapping.export.odin_gen.generate import (
    GeneratedCore,
    generate_core,
    write_generated_core,
)
from mimarsinan.mapping.export.odin_gen.render import spec_flags_word
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec, require_generatable
from mimarsinan.mapping.export.odin_gen.templates import render

KERNEL_TEMPLATE = "odin_gen_kernel.v"

#: The module name is the STOCK kernel's, deliberately: the frozen Vitis wrapper
#: `hw/fpga/kernel/odin_fpga_kernel_top.v` instantiates `odin_fpga_kernel` by
#: name, so a chip configuration selects its fabric by choosing which file
#: declares that module. The two are never compiled together.
KERNEL_MODULE = "odin_fpga_kernel"

#: The stock kernel file this one stands in for, named so a reader of either
#: knows the other exists.
STOCK_KERNEL_PATH = "hw/fpga/kernel/odin_fpga_kernel.v"


class FabricError(ValueError):
    """A spec cannot be the core of a fabric the frozen wrapper can drive."""


@dataclass(frozen=True)
class GeneratedFabric:
    """One chip configuration's generated RTL: the core and its kernel."""

    spec: CoreSpec
    core: GeneratedCore
    files: Tuple[Tuple[str, bytes], ...]

    def source_text(self, name: str) -> str:
        for filename, payload in self.files:
            if filename == name:
                return payload.decode("utf-8")
        raise KeyError(
            f"{name!r} is not one of this fabric's files: "
            f"{', '.join(filename for filename, _ in self.files)}")


def kernel_substitution_table(spec: CoreSpec) -> Dict[str, object]:
    """Every placeholder the kernel template declares, from the spec alone."""
    require_generatable(spec)
    return {
        "AXONS": spec.max_axons,
        "NEURONS": spec.max_neurons,
        "AW": spec.axon_address_bits,
        "NW": spec.neuron_address_bits,
        "MBITS": spec.membrane_bits,
        "WBITS": spec.weight_bits,
        "FLAGS": spec_flags_word(spec),
    }


def render_variant_kernel(spec: CoreSpec) -> str:
    """The variant kernel's Verilog, expanded from ``hw/gen`` for this spec."""
    return render(KERNEL_TEMPLATE, kernel_substitution_table(spec))


def kernel_filename() -> str:
    """The emitted file's name; the MODULE name is what the wrapper binds."""
    return f"{KERNEL_MODULE}.v"


def generate_fabric(spec: CoreSpec) -> GeneratedFabric:
    """Emit the core this spec declares AND the kernel that drives it.

    The STOCK spec has no generated fabric at all: its kernel is the committed
    file the routed bitstream was built from, and generating a second copy of it
    would be a second source of truth for RTL nobody would rebuild from here.
    """
    core = generate_core(spec)
    if core.vendored:
        raise FabricError(
            f"the stock spec ({spec.spec_key()}) is a vendored passthrough and "
            f"its kernel is the committed {STOCK_KERNEL_PATH}; there is no "
            f"fabric to generate for it. Select the stock chip configuration "
            f"instead of asking the generator for one.")
    kernel = render_variant_kernel(spec).encode("utf-8")
    return GeneratedFabric(
        spec=spec, core=core,
        files=tuple(core.files) + ((kernel_filename(), kernel),),
    )


def write_generated_fabric(fabric: GeneratedFabric, directory: Path
                           ) -> Tuple[Path, ...]:
    """Write the fabric's files and the core's descriptor under ``directory``."""
    root = Path(directory)
    written = list(write_generated_core(fabric.core, root))
    kernel = root / kernel_filename()
    kernel.write_bytes(fabric.source_text(kernel_filename()).encode("utf-8"))
    written.append(kernel)
    return tuple(written)
