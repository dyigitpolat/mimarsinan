#!/usr/bin/env python3
"""The v++ link configuration for N compute units, derived from the card's own.

ROUTE D. The card power session could not separate one ODIN compute unit from
kernel idle: the board reading moves in input-current steps of 85 to 110 mW at
12 V, above the unit's estimated dynamic power. N compute units of the SAME
kernel, driven by the same campaign at once, put N kernels' worth of activity
into the card-boundary difference. The RTL does not change; only the link does.

N IS DATA, AND IT IS THE ONLY INPUT. Everything else is read off the card's
config (``scripts/hacc/odin_<card>.cfg``, named by ``scripts/hacc/cards.sh``):

  * the kernel and the unit's name come from its one ``nk=`` line
    (``nk=odin_fpga_kernel_top:1:odin_0``): units are named ``odin_0`` to
    ``odin_<N-1>``, the base unit's name with its index replaced;
  * the master port, the memory family and the first bank come from that unit's
    ``sp=`` line (``sp=odin_0.m_axi_gmem:HBM[0]``): unit k gets bank
    (first bank + k) of the same family, so every unit has a bank of its own and
    no two units' buffers share a memory controller;
  * the number of banks the card has is declared in its profile
    (``memory_banks`` in ``cards.sh``) and passed in; a count past it refuses;
  * optional ``slr=`` placements (``--slrs SLR0,SLR0,...``, one per unit) are
    data too, computed outside (the route D fit check), never chosen here.

At N = 1 with no placements the output IS the card's config, byte for byte, so
the archived build stays reproducible from the same function. At N > 1 the
generated ``nk=`` and ``sp=`` lines replace the base ones at their own place,
one comment line says where they came from, and every other line is kept.

    python3 scripts/hacc/gen_link_cfg.py --base scripts/hacc/odin_u55c.cfg \\
        --compute-units 10 --memory-banks 32 --output build/hacc/hw_nc1_cu10/link_cu10.cfg

Standard library only: it runs on the build node, which has no ``src/``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

_NK = re.compile(r"^nk=(?P<kernel>[A-Za-z_][\w]*):(?P<count>\d+):(?P<names>[\w.]+)\s*$")
_SP = re.compile(r"^sp=(?P<unit>[\w]+)\.(?P<port>[\w]+):(?P<family>[A-Za-z]+)\[(?P<bank>\d+)\]\s*$")
_SLR = re.compile(r"^SLR\d$")
_INDEXED = re.compile(r"^(?P<prefix>.*?)_(?P<index>\d+)$")


class LinkConfigRefusal(ValueError):
    """The request cannot be linked as asked; the message says why."""


def _base_lines(base: str) -> Tuple[int, re.Match, int, re.Match]:
    nks = [(i, m) for i, line in enumerate(base.splitlines()) if (m := _NK.match(line))]
    if len(nks) != 1:
        raise LinkConfigRefusal(
            f"the base config carries {len(nks)} nk= line(s); a route D link is derived "
            f"from exactly one kernel's one compute unit")
    nk_index, nk = nks[0]
    names = nk.group("names").split(".")
    if int(nk.group("count")) != 1 or len(names) != 1:
        raise LinkConfigRefusal(
            f"the base config already declares {nk.group('count')} compute unit(s) "
            f"({nk.group('names')}); the base must be the one-unit config N is applied to")
    unit = names[0]
    sps = [(i, m) for i, line in enumerate(base.splitlines())
           if (m := _SP.match(line)) and m.group("unit") == unit]
    if len(sps) != 1:
        raise LinkConfigRefusal(
            f"the base config maps {len(sps)} port(s) of {unit} with sp=; the derivation "
            f"needs exactly one (the unit's one AXI master and its bank)")
    sp_index, sp = sps[0]
    return nk_index, nk, sp_index, sp


def compute_unit_names(base: str, units: int) -> List[str]:
    """The N unit names: the base unit's name with its index replaced by 0..N-1."""
    _nk_index, nk, _sp_index, _sp = _base_lines(base)
    unit = nk.group("names")
    match = _INDEXED.match(unit)
    prefix = match.group("prefix") if match else unit
    return [f"{prefix}_{k}" for k in range(int(units))]


def link_config(base: str, *, units: int, banks: int,
                slrs: Optional[Sequence[str]] = None) -> str:
    """The base config with its [connectivity] lines for ``units`` compute units."""
    units = int(units)
    if units < 1:
        raise LinkConfigRefusal(f"{units} compute units: a link needs at least one")
    nk_index, nk, sp_index, sp = _base_lines(base)
    first = int(sp.group("bank"))
    if first + units > int(banks):
        raise LinkConfigRefusal(
            f"{units} compute units from {sp.group('family')}[{first}] need banks up to "
            f"{sp.group('family')}[{first + units - 1}], and the card declares {int(banks)} "
            f"(memory_banks in scripts/hacc/cards.sh); one bank per unit is the rule")
    slrs = list(slrs) if slrs else []
    if slrs and len(slrs) != units:
        raise LinkConfigRefusal(
            f"{len(slrs)} SLR placement(s) for {units} compute units: give one per unit "
            f"or none")
    for slr in slrs:
        if not _SLR.match(slr):
            raise LinkConfigRefusal(f"{slr!r} is not an SLR name (SLR0, SLR1, ...)")
    if units == 1 and not slrs:
        return base
    names = compute_unit_names(base, units)
    kernel, port, family = nk.group("kernel"), sp.group("port"), sp.group("family")
    generated = [
        f"# generated by scripts/hacc/gen_link_cfg.py: {units} compute unit(s) of "
        f"{kernel}, unit k on {family}[{first} + k]"
        + (", placed by data (--slrs)" if slrs else ""),
        f"nk={kernel}:{units}:{'.'.join(names)}",
    ]
    generated += [f"sp={name}.{port}:{family}[{first + k}]" for k, name in enumerate(names)]
    generated += [f"slr={name}:{slr}" for name, slr in zip(names, slrs)]
    lines = base.splitlines()
    out: List[str] = []
    for index, line in enumerate(lines):
        if index == nk_index:
            out.extend(generated)
        elif index == sp_index:
            continue
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if base.endswith("\n") else "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="the card's one-unit v++ config")
    parser.add_argument("--compute-units", type=int, required=True)
    parser.add_argument("--memory-banks", type=int, required=True,
                        help="the card's bank count (memory_banks in cards.sh)")
    parser.add_argument("--slrs", default="",
                        help="comma-separated SLR placement, one per unit (optional)")
    parser.add_argument("--output", required=True)
    options = parser.parse_args(argv)
    try:
        with open(options.base, "r", encoding="utf-8") as handle:
            base = handle.read()
        text = link_config(base, units=options.compute_units, banks=options.memory_banks,
                           slrs=[s for s in options.slrs.split(",") if s] or None)
    except (OSError, LinkConfigRefusal) as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(options.output)), exist_ok=True)
    with open(options.output, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"[link-cfg] wrote {options.output}: {options.compute_units} compute unit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
