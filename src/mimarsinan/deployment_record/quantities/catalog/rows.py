"""The quantity catalog's rows: every priceable multiplicand, declared once with its dimension, unit and meaning."""

from __future__ import annotations

from typing import Tuple

from mimarsinan.deployment_record.units import DATA, DIMENSIONLESS, TIME

_COUNT = (DIMENSIONLESS, "1")

#: (key, (dimension, unit), doc) — declaration order is report order.
ROWS: Tuple[Tuple[str, Tuple[str, str], str], ...] = (
    # --- structural: the chip and what the mapping occupies ---------------------
    ("cells_used", _COUNT,
     "Weight cells the mapping actually occupies, post-compaction, as mapped."),
    ("cells_physical", _COUNT,
     "Weight cells the DECLARED chip provides (its parameter capacity over every "
     "declared core) — area prices the chip, not the cores a mapping allocated."),
    ("cores_physical", _COUNT,
     "Cores the DECLARED chip provides (the whole-chip static-power and area "
     "multiplicand)."),
    ("cores_allocated", _COUNT,
     "Hard cores the mapping allocates."),
    ("neurons_used", _COUNT, "Neuron slots the mapping drives."),
    ("neurons_physical", _COUNT,
     "Neuron slots the DECLARED chip provides, over every declared core."),
    ("axons_used", _COUNT, "Axon (row) slots the mapping drives."),
    ("axons_physical", _COUNT,
     "Axon (row) slots the DECLARED chip provides, over every declared core."),
    ("tiles", _COUNT, "NoC tiles of the resolved floorplan."),
    ("weight_bits", (DIMENSIONLESS, "bit"), "Declared weight precision of the run."),
    ("macs", _COUNT,
     "As-mapped MAC sites: every mapped instance's cells count, so a weight bank "
     "replicated across cores counts once per replica — the ENERGY multiplicand, "
     "because replicas really fire. Distinct from the logical census total_macs."),
    # --- program: passes, payloads, barriers -------------------------------------
    ("pass_count", _COUNT, "Passes of the deployed schedule."),
    ("compute_op_count", _COUNT,
     "Host ComputeOp invocations per inference — the per-invocation host "
     "overhead multiplicand [R2]: the measured host wall is dispatch-bound, "
     "not MAC-bound (~2000x past any rate model on the MLP study)."),
    ("sync_count", _COUNT, "Chip-wide synchronization barriers of the schedule."),
    ("reprogram_passes", _COUNT, "Passes that reprogram core weights."),
    ("reprogrammed_bytes", (DATA, "B"),
     "Programming payload bytes of the passes the program does not keep resident: on a sealed record its segments "
     "marked reprogram, on a candidate every pass the residency law does not keep (one count on both planes). A "
     "schedule fact that no pricer reads: the load law prices its own period census, which keeps a lone written "
     "pass loaded across rounds, so a round can rewrite less than this count."),
    ("connectivity_entries", _COUNT,
     "Connectivity (axon source span) entries of a sealed record's segments marked reprogram (their sealed totals); "
     "the candidate plane counts none. A schedule fact that no pricer reads: the load law counts the entries of "
     "exactly the cores its own census writes."),
    # [U-ENERGY class 3] Core-local memory traffic per inference, reads and writes apart.
    ("local_mem_read_bytes", (DATA, "B"),
     "Bytes the cores' local memories read per inference: each evaluated core reads every used row's input into its "
     "array once per evaluation step, at the program's value width (one bit per step on a spiking core)."),
    ("local_mem_write_bytes", (DATA, "B"),
     "Bytes the cores' local memories write per inference: each evaluated core writes every used column's output "
     "once per evaluation step, at the program's value width (one bit per step on a spiking core); a partial-sum "
     "core's outputs at its accumulator width are the named residual."),
    ("segment_cores", _COUNT,
     "Sum of core counts over EVERY pass — each pass resets its cores' neuron state, "
     "resident or not (the core-init multiplicand)."),
    ("reprogrammed_cores", _COUNT,
     "Cores holding a static program in the passes the program does not keep resident (the rule of "
     "reprogrammed_bytes, on both planes): a schedule fact that no pricer reads, the load law counting the cores its "
     "own census writes."),
    ("carried_raster_bytes", (DATA, "B"),
     "Payload HELD across INTRA-SEGMENT pass boundaries per inference, sized "
     "under the run's sealed transfer discipline: raster bits under verbatim, "
     "window counts under collapse. This is the buffer requirement, NOT the "
     "transfer — what crossed the chip boundary is carry_out_bytes / "
     "carry_in_bytes, dense under both disciplines. A pass that halves cores "
     "but doubles this is not a free win."),
    ("carry_out_bytes", (DATA, "B"),
     "Payload the chip EMITS across intra-segment pass boundaries per inference. "
     "Dense over the window under both disciplines: the collapse to window counts "
     "is done host-side, after the trace has left the chip."),
    ("carry_in_bytes", (DATA, "B"),
     "Payload the host RE-INJECTS across the same boundaries per inference. The "
     "chip's input interface takes spike trains, so this direction is dense too — "
     "and it rides the DMA channel that programming payloads ride."),
    ("carry_onchip_bytes", (DATA, "B"),
     "[U-F4] Byte accesses of a DECLARED on-chip carry buffer per inference: the values a wavefront group "
     "produces and reads itself, each written into the buffer once and read out once (both accesses counted; their "
     "crossings leave carry_out_bytes / carry_in_bytes). Absent without a declared carry_buffer_bytes."),
    ("carry_buffer_capacity_bytes", (DATA, "B"),
     "[U-F4] The DECLARED capacity of the on-chip carry buffer (carry_buffer_bytes): a chip component that leaks "
     "and occupies silicon whether or not a program fills it, so its static power and area are priced on it. Absent "
     "without the declaration."),
    # --- the write law (E49 period census at the declared batch K) ---------------------
    ("period_rewrite_bytes", (DATA, "B"),
     "Payload rewritten once per schedule period (K inferences): 0 when the programmed tiles "
     "fit the declared cores of their core type or every pass stays resident, else the payload "
     "of the cores the residency plan does not keep. Present only when inference_batch_K is declared."),
    ("runtime_written_cells", _COUNT,
     "Cells written per inference during deployment: backing-store writes / K plus "
     "producer-sourced writes."),
    ("written_units", _COUNT,
     "Write units per inference at the declared write granularity (cells, rows, columns or "
     "cores)."),
    ("write_events", _COUNT,
     "Write steps per inference: per pass, ceil(units / (write_units_in_parallel x "
     "write_ports))."),
    ("max_writes_per_cell_per_inference", _COUNT,
     "The most-written cell's writes per inference: its physical core's writes per period / K "
     "(every pass that rewrites the core writes it), plus one per inference where a producer "
     "writes cells (the high-wear end: the census places producer cells on no core): the endurance "
     "multiplicand. Absent where the plane places no rewritten core."),
    ("carry_peak_live_bytes", (DATA, "B"),
     "Peak concurrently-live carried payload over the segment's boundaries — the "
     "buffer a scheduled chip must provision. Wires with disjoint live ranges "
     "share it. A capacity fact for constraints; deliberately not an energy "
     "multiplicand."),
    # --- dynamics: what an execution did ------------------------------------------
    ("timesteps", _COUNT, "Timesteps of one inference window (S)."),
    ("neuron_cycles_per_window", _COUNT, "[U-LIF2 W12] Cycles a neuron runs per window: S, or 2S where a synchronized "
     "LIF neuron integrates then emits. The neuron-update and leak multiplicand."),
    ("latency_steps", _COUNT,
     "End-to-end latency in timesteps: the sum of per-segment executed timesteps "
     "for one sample — what t_cycle converts to seconds."),
    # [thru:w3] An input's span under the batch pipeline (batch_pipeline_slots): per input, never per round.
    ("input_span_steps", _COUNT,
     "[thru:w3] Under the batch pipeline, the executed steps of the passes from an input's first pass to its last "
     "(the largest over the run's inputs): the passes it waits across, other inputs' instances in them included."),
    ("input_span_syncs", _COUNT,
     "[thru:w3] The barriers between the passes of an input's span (the largest over the run's inputs)."),
    ("input_span_cores", _COUNT,
     "[thru:w3] The cores the passes of an input's span evaluate, each pass's once (the largest over the run's "
     "inputs): their initializations lie on the input's path."),
    ("total_spikes", _COUNT,
     "Spike EMISSIONS counted by the simulator — not synapse arrivals."),
    ("boundary_events", _COUNT,
     "Value/spike events crossing observed stage boundaries (gate-reduced totals)."),
    ("synaptic_events", _COUNT,
     "Spike arrivals at active synapses — one event is one MAC on an event-driven "
     "chip. No producer seals this yet: at candidate time it is MODELED from the "
     "declared activity factor; at record time it is absent until an event census "
     "is sealed."),
    ("synaptic_events_busiest_core", _COUNT,
     "[U-C2] Per (pass, segment) row, the synaptic events of its busiest physical core, summed over the rows: the "
     "multiplicand of an event-serial core's wall (t_synaptic_op) on a chip of several concurrent cores, whose cores "
     "of one row run at once and whose rows run in turn. MODELED at candidate time by the energy's own model (a "
     "core's as-mapped cells x timesteps x the declared activity_factor; the value rule's one event per cell), and "
     "formed only where the resolved physics prices compute per synaptic operation; no record states it, since a "
     "sealed event census is one total per inference."),
    ("noc_total_packets", _COUNT, "All NoC packets of the execution."),
    ("noc_intra_tile_packets", _COUNT, "Packets delivered within their source tile."),
    ("noc_inter_tile_packets", _COUNT, "Packets that left their source tile."),
    ("noc_total_hops", _COUNT,
     "Total link traversals: the sum of per-link packet counts over the NoC "
     "(derived from the sealed link loads; one packet crossing three links is "
     "three hops)."),
    ("noc_value_bytes", (DATA, "B"),
     "Value-domain NoC traffic per inference, in byte-hops: every inter-core wire of an "
     "execution stage carries one value of activation_bits / 8 bytes over its XY hops."),
    # --- compute lanes: produced by a capability census (none by default) ---------------
    ("lane_ops", _COUNT, "Element operations a program runs on arithmetic lanes per inference."),
    ("lanes_per_tile", _COUNT, "Output arithmetic lanes per tile at the program's capability level."),
    # --- periphery: producers land with the conversion models (C6) ----------------
    ("adc_conversions", _COUNT,
     "Analog-to-digital conversions of the execution, produced by the target's "
     "declared conversion model; identically zero on fully digital targets."),
    ("adc_count", _COUNT,
     "ADC instances the chip needs (columns / sharing factor), produced by the "
     "target's declared conversion model."),
    # --- host / the logical split --------------------------------------------------
    ("host_ops_s", (TIME, "s"),
     "Measured wall time of host ComputeOps per inference on the development machine (stage_timing.host_wall, "
     "amortized): a diagnostic no pricer reads [U-HOST]."),
    ("total_params", _COUNT, "Logical parameter census of the deployed model."),
    ("onchip_params", _COUNT, "Logical parameters realized on chip cores."),
    ("host_params", _COUNT, "Logical parameters held by host ComputeOps."),
    ("total_macs", _COUNT,
     "Logical forward-MAC census of one inference — replication-free, the host/chip "
     "SPLIT census (distinct from the as-mapped energy multiplicand macs)."),
    ("onchip_macs", _COUNT, "Logical forward MACs realized on chip."),
    ("host_macs", _COUNT, "Logical forward MACs executed host-side."),
    # [U-J] The host-chip boundary beside the pass carry, and the host's memory-bound work (formed where the
    # resolved physics prices the host-chip link: an edge host block).
    ("segment_in_bytes", (DATA, "B"), "Bytes the host sends the chip per inference at segment boundaries: the network "
     "input read on chip and every host operator's output read on chip, whole slices, at the carried-value grid."),
    ("segment_out_bytes", (DATA, "B"), "Bytes the chip sends the host per inference at segment boundaries: every "
     "on-chip slice a host operator reads and the readout, at the carried-value grid."),
    ("host_link_transfers", _COUNT, "Host-chip transfers per inference: one per direction at each segment boundary "
     "and each pass boundary that carries values to a store the chip does not own [U-J3]."),
    ("host_memory_bound_bytes", (DATA, "B"), "Bytes the MAC-free host operators (element-wise, pooling) read and write "
     "per inference, at the host executor's float32: the work memory binds."),
)
