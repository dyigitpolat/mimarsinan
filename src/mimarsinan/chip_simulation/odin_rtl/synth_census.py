"""The Xilinx cell census: primitives into resource columns, memories into BRAM or not."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

NEURON_MEMORY_MODULE = "SRAM_256x128_wrapper"
SYNAPSE_MEMORY_MODULE = "SRAM_8192x32_wrapper"
MEMORY_MODULES: Tuple[str, ...] = (NEURON_MEMORY_MODULE, SYNAPSE_MEMORY_MODULE)

#: Xilinx primitive families, by cell-name prefix. `INV` is a LUT1 in disguise
#: once Vivado packs it, so it is counted apart and added back in the report.
BRAM_PREFIXES: Tuple[str, ...] = ("RAMB",)
URAM_PREFIXES: Tuple[str, ...] = ("URAM",)
#: Distributed RAM and SRL cells: the memory did NOT reach a block-RAM tile.
DISTRIBUTED_RAM_PREFIXES: Tuple[str, ...] = ("RAM", "SRL")
FLIP_FLOP_PREFIXES: Tuple[str, ...] = ("FD",)
CARRY_PREFIXES: Tuple[str, ...] = ("CARRY",)
MUXF_PREFIXES: Tuple[str, ...] = ("MUXF",)
IO_PREFIXES: Tuple[str, ...] = ("IBUF", "OBUF", "IOBUF", "BUF")

_LUT_CELL = re.compile(r"^LUT[1-9]$")
_MEMORY_GEOMETRY = re.compile(r"^SRAM_(\d+)x(\d+)_wrapper$")


class SynthesisError(RuntimeError):
    """The synthesis did not run, or did not report what the gate reads."""


def _has_prefix(cell: str, prefixes: Sequence[str]) -> bool:
    return any(cell.startswith(prefix) for prefix in prefixes)


def _is_block_ram(cell: str) -> bool:
    return _has_prefix(cell, BRAM_PREFIXES) or _has_prefix(cell, URAM_PREFIXES)


def _is_distributed_ram(cell: str) -> bool:
    return not _is_block_ram(cell) and _has_prefix(cell, DISTRIBUTED_RAM_PREFIXES)


def _selected(cells: Mapping[str, int], predicate) -> Dict[str, int]:
    return {cell: count for cell, count in cells.items() if predicate(cell)}


def _total(cells: Mapping[str, int], predicate) -> int:
    return sum(count for cell, count in cells.items() if predicate(cell))


@dataclass(frozen=True)
class ResourceTable:
    """One ODIN core's primitive census, in the columns a resource budget uses."""

    luts: int
    inverters: int
    flip_flops: int
    carry: int
    muxf: int
    bram36: int
    bram18: int
    uram: int
    lutram_cells: Mapping[str, int]
    io_buffers: int
    unclassified: Mapping[str, int]
    total_cells: int

    @property
    def bram_tiles(self) -> float:
        """RAMB18 is half a physical tile; RAMB36 is one."""
        return self.bram36 + self.bram18 / 2.0

    @property
    def lutram(self) -> int:
        """Distributed-RAM/SRL cells: memory that stayed in the CLB fabric."""
        return sum(self.lutram_cells.values())

    def as_record(self) -> Dict[str, Any]:
        return {
            "luts": self.luts,
            "inverters": self.inverters,
            "lut_equivalent": self.luts + self.inverters,
            "flip_flops": self.flip_flops,
            "carry": self.carry,
            "muxf": self.muxf,
            "bram36": self.bram36,
            "bram18": self.bram18,
            "bram_tiles": self.bram_tiles,
            "uram": self.uram,
            "lutram": self.lutram,
            "lutram_cells": dict(sorted(self.lutram_cells.items())),
            "io_buffers": self.io_buffers,
            "unclassified": dict(sorted(self.unclassified.items())),
            "total_cells": self.total_cells,
        }


def resource_table(cells_by_type: Mapping[str, int]) -> ResourceTable:
    """Partition a primitive census into the resource columns; nothing is dropped."""
    claimed = {
        "lut": lambda c: bool(_LUT_CELL.match(c)),
        "inv": lambda c: c == "INV",
        "ff": lambda c: _has_prefix(c, FLIP_FLOP_PREFIXES),
        "carry": lambda c: _has_prefix(c, CARRY_PREFIXES),
        "muxf": lambda c: _has_prefix(c, MUXF_PREFIXES),
        "bram36": lambda c: c.startswith("RAMB36"),
        "bram18": lambda c: c.startswith("RAMB18"),
        "uram": lambda c: _has_prefix(c, URAM_PREFIXES),
        "lutram": _is_distributed_ram,
        "io": lambda c: _has_prefix(c, IO_PREFIXES),
    }
    unclassified = _selected(
        cells_by_type,
        lambda c: not any(predicate(c) for predicate in claimed.values()))
    return ResourceTable(
        luts=_total(cells_by_type, claimed["lut"]),
        inverters=_total(cells_by_type, claimed["inv"]),
        flip_flops=_total(cells_by_type, claimed["ff"]),
        carry=_total(cells_by_type, claimed["carry"]),
        muxf=_total(cells_by_type, claimed["muxf"]),
        bram36=_total(cells_by_type, claimed["bram36"]),
        bram18=_total(cells_by_type, claimed["bram18"]),
        uram=_total(cells_by_type, claimed["uram"]),
        lutram_cells=_selected(cells_by_type, claimed["lutram"]),
        io_buffers=_total(cells_by_type, claimed["io"]),
        unclassified=unclassified,
        total_cells=sum(cells_by_type.values()),
    )


def _stat_section(stat: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = stat.get(key)
    if not isinstance(section, dict):
        raise SynthesisError(
            f"the yosys stat document has no {key!r} object; the gate reads a "
            f"`stat -json` census and got keys {sorted(stat)}")
    return section


def primitive_cells(stat: Mapping[str, Any]) -> Dict[str, int]:
    """The hierarchical cell census with module INSTANCES removed."""
    modules = {name.lstrip("\\") for name in _stat_section(stat, "modules")}
    census = _stat_section(_stat_section(stat, "design"), "num_cells_by_type")
    return {cell: int(count) for cell, count in census.items()
            if cell.lstrip("\\") not in modules}


@dataclass(frozen=True)
class MemoryInference:
    """What the synthesizer did with one memory wrapper: block RAM, or not."""

    module: str
    present: bool
    words: int
    width: int
    bram_cells: Mapping[str, int]
    distributed_cells: Mapping[str, int]
    flip_flops: int
    cells: Mapping[str, int]

    @property
    def inferred_as_bram(self) -> bool:
        return self.present and bool(self.bram_cells) and not self.distributed_cells

    @property
    def geometry(self) -> str:
        return f"{self.words}x{self.width}"

    def finding(self) -> str | None:
        """The LOUD text when the memory did not reach a block-RAM tile."""
        if self.inferred_as_bram:
            return None
        head = (f"FINDING (plan §7 row 19): the {self.geometry} memory "
                f"`{self.module}` did NOT infer block RAM")
        if not self.present:
            return (f"{head} -- yosys kept no `{self.module}` module at all, so the "
                    f"BRAM overlay in hw/fpga/mem/ was not the declaration that won "
                    f"the source order, or the hierarchy was flattened away.")
        if self.distributed_cells:
            return (f"{head} -- yosys mapped it to distributed RAM/SRL cells "
                    f"{dict(sorted(self.distributed_cells.items()))}. On a U55C "
                    f"that costs LUTs the core budget does not have; report it, "
                    f"do not absorb it.")
        return (f"{head} -- no RAMB/URAM cell in the module; its census is "
                f"{dict(sorted(self.cells.items()))} ({self.flip_flops} flip-flops), "
                f"i.e. the array degraded to registers or logic.")

    def as_record(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "geometry": self.geometry,
            "words": self.words,
            "width": self.width,
            "present": self.present,
            "bram_cells": dict(sorted(self.bram_cells.items())),
            "distributed_cells": dict(sorted(self.distributed_cells.items())),
            "flip_flops": self.flip_flops,
            "inferred_as_bram": self.inferred_as_bram,
        }


def _memory_geometry(module: str) -> Tuple[int, int]:
    match = _MEMORY_GEOMETRY.match(module)
    if match is None:
        raise SynthesisError(
            f"cannot read a words x width geometry out of the memory module name "
            f"{module!r}; the wrappers are named SRAM_<words>x<width>_wrapper")
    return int(match.group(1)), int(match.group(2))


def memory_inferences(stat: Mapping[str, Any]) -> Tuple[MemoryInference, ...]:
    """What the synthesizer did with each of the two memory wrappers."""
    modules = _stat_section(stat, "modules")
    inferences: List[MemoryInference] = []
    for module in MEMORY_MODULES:
        words, width = _memory_geometry(module)
        body = modules.get(f"\\{module}", modules.get(module))
        cells: Dict[str, int] = (
            {} if not isinstance(body, dict)
            else {cell: int(count) for cell, count
                  in _stat_section(body, "num_cells_by_type").items()})
        inferences.append(MemoryInference(
            module=module,
            present=isinstance(body, dict),
            words=words,
            width=width,
            bram_cells=_selected(cells, _is_block_ram),
            distributed_cells=_selected(cells, _is_distributed_ram),
            flip_flops=_total(cells, lambda c: _has_prefix(c, FLIP_FLOP_PREFIXES)),
            cells=cells,
        ))
    return tuple(inferences)
