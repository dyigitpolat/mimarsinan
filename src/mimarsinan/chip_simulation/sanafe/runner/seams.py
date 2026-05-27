"""Patchable SANA-FE integration entry points for ``SanafeRunner`` tests."""

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_numpy
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import is_ttfs_spiking_mode
from mimarsinan.chip_simulation.sanafe.arch_synth import _sanafe, build_architecture, derive_arch_spec
from mimarsinan.chip_simulation.sanafe.net_synth import (
    build_network_for_segment,
    set_always_on_spike_trains,
    set_input_spike_trains,
    set_ttfs_input_spike_trains,
)
