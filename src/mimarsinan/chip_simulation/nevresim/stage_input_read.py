"""[W33] Where nevresim's sources read a stage input in each core's local time: the declaration the stage-input read
probe reads (``chip_simulation/stage_input_clock``).

nevresim's generated simulator reads every stage-input row at the reading core's local cycle only where its sources
carry W33's change (``RA/11_wave2/exact/nevresim.patch``, base 90af7d2). nevresim is a nested repository, and the
commit of that change waits for its owner, so a checkout can hold either tree. Base 90af7d2 reads every row at the
global cycle: a core above latency 0 reads its row that many cycles early and never its last cycles, with no error.
The guard therefore counts nevresim among the backends that read in core-local time only where a probe of the tree it
compiles against finds every read site below and the tree's own self-test holds."""

from __future__ import annotations

from typing import Tuple

from mimarsinan.chip_simulation.nevresim.nevresim_driver import resolve_nevresim_path

#: One read site: its path under ``include/``, the text the site holds, and the global-cycle reads it replaced.
ReadSite = Tuple[str, str, Tuple[str, ...]]

#: The read: the train every core reads in its own local time (``read_stage_input``), the compute policy's input-span
#: and network-output reads through it, the two cycle-timed executions that build the train once per sample (they
#: generated one input vector per global cycle), and the self-test.
STAGE_INPUT_READ_SITES: Tuple[ReadSite, ...] = (
    ("simulator/spike_generation/stage_input_train.hpp", "read_stage_input(", ()),
    ("simulator/compute_policy/compute_policy_base.hpp", "read_stage_input(",
     ("input_buffer[span.start_ + i]", "input_buffer[source.neuron_]")),
    ("simulator/execution/spiking_execution.hpp", "StageInputTrain<", ("generate_spikes(input, i)",)),
    ("simulator/execution/ttfs_execution.hpp", "StageInputTrain<", ("generate_spikes(input, i)",)),
    ("_tests/stage_input_clock.hpp", "test_stage_input_read_in_core_local_time", ()),
)

#: The tree's own self-test of the read, (header, constexpr function): two one-neuron cores at latencies 0 and 2 read
#: one front-loaded train through ``SpikingExecution`` and fire 2 and 2 times, which only a core-local read gives (read
#: at the global cycle, the second fires 0 times).
SELF_TEST: Tuple[str, str] = ("_tests/stage_input_clock.hpp",
                              "nevresim::tests::test_stage_input_read_in_core_local_time")


def nevresim_tree() -> str:
    """The nevresim tree a run compiles against: ``NevresimDriver.nevresim_path``, else ``nevresim_root()``
    (``resolve_nevresim_path``)."""
    return str(resolve_nevresim_path())
