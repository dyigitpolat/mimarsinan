#!/usr/bin/env python3
"""Emit the Vitis kernel.xml for the ODIN RTL kernel from the host-side SSOT.

The argument ids, offsets and types are the ones
``mimarsinan.chip_simulation.odin_fpga.xrt_transport`` calls the kernel with, so
the packaged kernel and the host that drives it cannot disagree about the
register map. Run it on the cluster (``scripts/hacc/build_xclbn.sh`` does).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimarsinan.chip_simulation.odin_fpga.xrt_transport import (  # noqa: E402
    ARG_CAPTURE,
    ARG_CAPTURE_WORDS,
    ARG_PROGRAM,
    ARG_PROGRAM_WORDS,
    ARG_STIMULUS,
    ARG_STIMULUS_WORDS,
    KERNEL_NAME,
)

#: (id, name, port, offset, size, hostSize, type) — the Vitis RTL-kernel
#: convention: 64-bit pointers at 0x10/0x1C/0x28, scalars after them.
ARGS = (
    (ARG_PROGRAM, "program", "m_axi_gmem", "0x10", "0x8", "0x8", "int*"),
    (ARG_STIMULUS, "stimulus", "m_axi_gmem", "0x1C", "0x8", "0x8", "int*"),
    (ARG_CAPTURE, "capture", "m_axi_gmem", "0x28", "0x8", "0x8", "int*"),
    (ARG_PROGRAM_WORDS, "program_words", "s_axi_control", "0x34", "0x4", "0x4", "uint"),
    (ARG_STIMULUS_WORDS, "stimulus_words", "s_axi_control", "0x3C", "0x4", "0x4", "uint"),
    (ARG_CAPTURE_WORDS, "capture_events", "s_axi_control", "0x44", "0x4", "0x4", "uint"),
)


def build(kernel_name: str) -> ET.ElementTree:
    root = ET.Element("root", versionMajor="1", versionMinor="6")
    ET.SubElement(root, "kernel",
                  name=kernel_name, language="ip_c",
                  vlnv=f"mimarsinan:kernel:{kernel_name}:1.0",
                  attributes="", preferredWorkGroupSizeMultiple="0",
                  workGroupSize="1", interrupt="true",
                  hwControlProtocol="ap_ctrl_hs")
    kernel = root.find("kernel")
    assert kernel is not None
    ports = ET.SubElement(kernel, "ports")
    ET.SubElement(ports, "port", name="s_axi_control", mode="slave",
                  range="0x1000", dataWidth="32", portType="addressable",
                  base="0x0")
    ET.SubElement(ports, "port", name="m_axi_gmem", mode="master",
                  range="0xFFFFFFFFFFFFFFFF", dataWidth="32",
                  portType="addressable", base="0x0")
    args = ET.SubElement(kernel, "args")
    for index, name, port, offset, size, host_size, type_name in ARGS:
        ET.SubElement(args, "arg", name=name, addressQualifier=(
            "1" if port == "m_axi_gmem" else "0"),
            id=str(index), port=port, size=size, offset=offset,
            type=type_name, hostOffset="0x0", hostSize=host_size)
    return ET.ElementTree(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel", default=KERNEL_NAME)
    options = parser.parse_args()
    tree = build(options.kernel)
    ET.indent(tree, space="  ")
    tree.write(options.output, encoding="utf-8", xml_declaration=True)
    print(f"[hacc-build] wrote {options.output} ({len(ARGS)} kernel arguments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
