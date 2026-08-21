"""``generate_core``: one ``CoreSpec`` becomes RTL, a descriptor, and its files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from mimarsinan.mapping.export.odin_gen.descriptor import build_descriptor
from mimarsinan.mapping.export.odin_gen.feasibility import SaturationBound
from mimarsinan.mapping.export.odin_gen.passthrough import (
    assert_stock_passthrough,
    is_stock_spec,
    vendored_file_set,
)
from mimarsinan.mapping.export.odin_gen.render import core_filename, render_core_rtl
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec, require_generatable

DESCRIPTOR_FILENAME = "core_descriptor.json"


@dataclass(frozen=True)
class GeneratedCore:
    """One core's emitted sources plus the descriptor that explains them."""

    spec: CoreSpec
    files: Tuple[Tuple[str, bytes], ...]
    descriptor: Dict[str, Any]
    vendored: bool

    def source_bytes(self, name: str) -> bytes:
        for filename, payload in self.files:
            if filename == name:
                return payload
        raise KeyError(
            f"{name!r} is not one of this core's files: "
            f"{', '.join(filename for filename, _ in self.files)}")

    def source_text(self, name: str) -> str:
        """The GENERATED source as text; vendored files are read as bytes."""
        return self.source_bytes(name).decode("utf-8")


def generate_core(
    spec: CoreSpec, *, saturation_bounds: Tuple[SaturationBound, ...] = (),
) -> GeneratedCore:
    """Emit the core this spec declares.

    The STOCK spec is a vendored PASSTHROUGH, asserted byte-identical to
    ``hw/vendor/odin``: the out-of-the-box deployment reads that tree directly
    and never depends on this function, so generation of the stock point is a
    proof that the generator agrees with the silicon, not a source of it.
    """
    if is_stock_spec(spec):
        files = vendored_file_set()
        assert_stock_passthrough(files)
        return GeneratedCore(
            spec=spec, files=files, vendored=True,
            descriptor=build_descriptor(
                spec, files=[name for name, _ in files], vendored=True),
        )
    require_generatable(spec)
    files = ((core_filename(spec), render_core_rtl(spec).encode("utf-8")),)
    return GeneratedCore(
        spec=spec, files=files, vendored=False,
        descriptor=build_descriptor(
            spec, files=[name for name, _ in files], vendored=False,
            saturation_bounds=saturation_bounds),
    )


def write_generated_core(core: GeneratedCore, directory: Path) -> Tuple[Path, ...]:
    """Write the emitted files and the descriptor under ``directory``."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in core.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        written.append(path)
    descriptor = root / DESCRIPTOR_FILENAME
    descriptor.write_text(
        json.dumps(core.descriptor, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    written.append(descriptor)
    return tuple(written)
