"""Registry entries: the physical ODIN backend and the device it is reached through."""

from __future__ import annotations

from mimarsinan.chip_simulation.odin_fpga.factory import TRANSPORTS
from mimarsinan.config_schema.registry.entries_platform_backends import (
    _meta_backend_enable,
    _why_backend_enable,
)
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_XRT = "xrt"


ENTRIES = (
    _E("enable_odin_fpga_simulation", domain="event", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="ODIN FPGA Deployment",
       doc="Whether the physical ODIN backend runs (export -> device -> counts, "
           "certified against the HCM reference). OPT-IN: the capability "
           "derivation admits it wherever the declared soma law has a crossbar "
           "executor, but it stays off until the document asks for it, because "
           "running it needs a device (an RTL simulator or an Alveo board) that "
           "no config can assume is present.",
       derived_from=("spiking_mode", "firing_granularity", "membrane_arithmetic"),
       why=_why_backend_enable(
           "odin_fpga",
           "the ODIN crossbar has no executor for this soma law or mode"),
       meta=_meta_backend_enable("odin_fpga"), provenance="ConversionPolicy recipe"),
    _E("odin_fpga_transport", domain="event", group="deployment_target",
       owner="odin_fpga_backend", type=T.ENUM, options=TRANSPORTS,
       category=Category.ADVANCED, exposure="user", label="ODIN Device Transport",
       effect="WHICH device executes the exported program",
       doc="rtl_cosim: the vendored RTL under a local simulator — the same "
           "program, no board needed. xrt: an Alveo card through the Xilinx "
           "runtime. The two are handed byte-identical program payloads; the "
           "board is a transport swap, not a second backend.",
       relevant=R.when_true("enable_odin_fpga_simulation")),
    _E("odin_fpga_sample_count", domain="event", group="deployment_target",
       owner="odin_fpga_backend", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="ODIN Sample Count",
       doc="Deterministic test samples run through the device.", bounds=(1, None),
       relevant=R.when_true("enable_odin_fpga_simulation")),
    _E("odin_fpga_xclbin_path", domain="event", group="deployment_target",
       owner="odin_fpga_backend", type=T.PATH, category=Category.ADVANCED,
       exposure="user", label="ODIN xclbin Path",
       doc="The built kernel image to load on the board (scripts/hacc/"
           "build_xclbn.sh produces it). Required by the xrt transport and "
           "unused by the cosimulation.",
       relevant=R.all_of(R.when_true("enable_odin_fpga_simulation"),
                         R.when("odin_fpga_transport", in_=(_XRT,)))),
    _E("odin_fpga_device_index", domain="event", group="deployment_target",
       owner="odin_fpga_backend", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="ODIN Device Index",
       doc="Which Alveo card on the node the session opens (0 = the first).",
       bounds=(0, None),
       relevant=R.all_of(R.when_true("enable_odin_fpga_simulation"),
                         R.when("odin_fpga_transport", in_=(_XRT,)))),
)
