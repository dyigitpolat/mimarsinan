# ODIN — platform physics profile

The 256-neuron, 64k-synapse online-learning core of Frenkel et al. (IEEE TBioCAS 2019):
28 nm FDSOI, one time-multiplexed SRAM crossbar, 320 µm × 270 µm. Measured on silicon,
averaged over **nine test chips at 24 °C**. The full research pass is
[`docs/research/physics/silicon_correlation_research.md`](../../../../../docs/research/physics/silicon_correlation_research.md).

This is the profile that **taught the framework the difference between a marginal
energy and an operating-point average**, because ODIN's paper publishes both and names
them separately.

## The one thing to understand about this target

ODIN publishes its own power model, and it is affine:

```
P = P_leak + P_idle x f_clk + E_SOP x r_SOP        (Eq. 2)
E_tot,SOP = P / r_SOP                              (Eq. 3)
```

with `P_leak = 27.3 µW`, `P_idle = 1.78 µW/MHz`, `E_SOP = 8.43 pJ` — and a headline
`E_tot,SOP > 12.7 pJ`. **The headline is Eq. (3), not a constant.** The paper is
explicit that it divides "the whole chip power consumption P … without subtracting
contributions from leakage and idle power", so it changes with the operating point:
12.7 pJ in accelerated time, **54 pJ** at biological rates, from the same silicon.

The profile therefore declares `e_mac = 8.43 pJ` (Eq. 2's incremental term) with
leakage and idle power as separate static constants. Priced that way, ODIN's two
published operating points — 13.6× apart in power — come out at **−0.02 %** and
**+0.27 %**. Priced with the 12.7 pJ headline instead, the biological point comes out
at **−76 %**. Both are in the correlation suite; the second is the regression test.

## Notes on individual constants

- **`e_mac` already contains the neuron update.** The paper says `E_SOP` "includes …
  reading and updating the associated neuron state, as well as the controller and
  scheduler overheads", so `e_neuron_update` is deliberately absent. Declaring it would
  charge the neuron twice.
- **`p_static_per_core` is the CLOCK-proportional idle term, not leakage.** It is
  `1.78 µW/MHz × 75 MHz` evaluated at this profile's declared clock. A run at a
  different clock must restate it — the biological-time reference case does exactly
  that, through an override that the correlation report prints.
- **ODIN has no timestep, so `t_cycle` is undeclared; its compute is priced per synaptic
  operation** (`t_synaptic_op`, the time twin of `e_mac`): "each SOP takes two clock cycles
  to complete", so 2 / 75 MHz = 26.667 ns per operation. One input event is one sweep of the
  core: "all N neurons of ODIN will be updated ... each neuron spike event thus leads to N
  SOPs", "256 SOPs = 512 clock cycles" (Fig. 3), and the RTL's `POP_NEUR` state runs those
  512 cycles whatever the mapping uses of the 256 neurons (`src/controller.v:208`). The
  latency, the throughput and the static energy are therefore priced on the event census
  (256 operations per input event, the whole sweep; a census that counts an arrival once per
  used column prices a partly used core low by its neuron occupancy, the energy too), and
  refused by name without a census or
  a declared `activity_factor`, as the energy is. The census total is one core's events. On a
  chip of several concurrent ODIN cores the cores of a pass run at once and the passes in turn,
  so the candidate plane prices each pass at its busiest core (its as-mapped cells x timesteps x
  `activity_factor`, the energy's own model, `synaptic_events_busiest_core`); a view that states
  the census as one total (a sealed record) refuses the latency by name on several cores. A
  `t_cycle` of one synaptic operation priced a timestep of the executed
  census at 26.67 ns, at least 256 times too fast for any step that carries an input event.
- **`area_per_cell` (0.68 µm²/synapse) is declared but never summed** with the
  aggregate `area_per_core_total`. It is the only per-cell area among the three
  correlated targets that can be checked against a published total: 65,536 × 0.68 µm²
  = 51.6 % of the core, leaving roughly half for the neurons, the SDSP update logic,
  the scheduler and the controller.

## The programming group — DERIVED, and how (TS6, U-C2)

Frenkel reports no weight-load energy, so the two programming energies are **authored
derivations** anchored on `e_mac` (`evidence_kind: derived`), each with its nominal at the
band's log-space centre (`sqrt(low x high)`). The programming wall is derived from ODIN's own
published SPI frame and core image (the authors' RTL and documentation):

- **`e_program_per_byte` = 8.43 / 20.649 / 50.58 pJ/B.** An ODIN synapse is 4 bits
  (3-bit weight + 1 mapping bit), so a byte of synaptic SRAM holds two synapses and the
  per-byte access anchor is `2 x e_mac` = 16.86 pJ/B. An SRAM write costs ~1–3× its
  read → band `[0.5x, 3x]`. `e_mac` already contains the weight read *and* update, the
  neuron state access and the controller overheads, so it overstates the bare SRAM
  access: that is what the 0.5× corner is for. It multiplies the occupied payload bytes,
  while a load writes the whole 36,864-byte image: the commit energy of the unoccupied
  image words is not priced.
- **`t_core_program` = 78.6432 ms per core load (U-C2).** A load writes the whole core
  image, whatever the program occupies: 8,192 32-bit synapse words and 256 128-bit neuron
  words, 36,864 byte transactions. One transaction is 40 SCK (`src/spi_slave.v:82-84`) and
  "SCK must be at least 4x slower than CLK" (`doc/README.md` Sec. 2.1), so a byte takes at
  least 160 clock cycles: 36,864 x 160 / 75 MHz = 78.6432 ms, a floor (a slower SPI master
  is a slower load). It is the time twin of `e_core_program`, read by `latency_programming_s`,
  the write law's per-pass walls, Pi, M, the regime labels and the write-bandwidth batch.
  The card's measured programming run, 6,345,255 cycles for the 36,864-byte image at
  100 MHz (172.13 cycles per byte, 7.6% above the floor), is printed beside it as a check,
  never as the declaration. No `t_program_per_byte` is declared: the donor-scaled
  5.09 to 1,272.73 ns per byte it replaces lay below the frame's own 2,133.3 ns per byte at
  every corner, and multiplied the occupied-cell bytes, a fraction of the image a load
  writes.
- **`e_core_program` = 1.079 / 2.158 / 4.316 nJ.** No `e_core_init` exists here, so the
  anchor is the **core-state sweep** the geometry defines: 256 neurons × `e_mac` =
  2.158 nJ. Band `[0.5x, 2x]`. ODIN is a single core, so a program load pays this once.

## What this profile deliberately does not declare

The whole `interconnect` group: ODIN is a **single core** with no NoC, so hop and
packet energies are not absent-for-lack-of-data, they are structurally zero — and a
multi-core deployment mapped onto "ODIN" would be a different chip, not this one.

Also absent: `t_cycle` (no fixed timestep; the compute is `t_synaptic_op` on the event
census), `t_program_per_byte` (a load writes the whole image; its wall is `t_core_program`),
the `host` group (a property of the deployment host), `e_dma_per_byte`
(the SPI transfer energy onto the chip is unpublished — only the *commit* above is
derived), `e_core_init` / `t_core_init` (no reset measurement exists),
`e_sync_barrier` / `t_sync_barrier` (nothing to synchronize across), and
`membrane_bits` / `area_per_state_bit` — the neuron state width is configurable across
the 20 Izhikevich behaviours and no single width is published.

## Scope warning

ODIN is a **single 256-neuron core**. It is in the library as a correlation anchor and
as a small-edge design point, not as a target for large workloads: a mapping that needs
more than 256 neurons or 64k synapses does not fit one ODIN, and the capacity
constraint — not this profile — is what must say so.

## Wave 1 rows: validity and endurance (G3)

- **Validity geometry.** One 256 x 256 core of 3-bit weights (the fourth synapse bit maps, it does not weigh):
  the per-core footprint `area_per_core_total` holds at that core only, so a chip declaring another core
  geometry, more cores, or weights wider than 3 bits refuses `chip_area_mm2` by name (E190).
- **`endurance_cycles` = 1e16** (published, ielmini2018inmemory p.7): ODIN's synapses are SRAM, so its lifetime is
  priced on the embedded-SRAM row, at **`pulses_per_write` = 1** (derived): one SRAM write is one write cycle.

## Chip Area by the Area Rule (`area_rule`)

[U-AREA] `chip_area_mm2` of this description is priced from its `area_rule` block (the area rule from published silicon per technology type, `platform_physics/area_rule`; design revision 3). Where the rows or notes above describe another chip-area path (a per-core footprint, `area_global_fixed`, or a refusal outside the footprint's validity), the block supersedes them for chip area; the other constants keep their uses.

- **Type and terms.** T1 (digital spiking cores); an anchor: its own published core scaled by the census's cells, converters or bits (rho), its size and word width flagged.
- **Outside part (`outside: type`).** the type table's core-grid share (gamma, flagged); an area_global_fixed beside it is refused by name ('the outside part is stated twice').
- **Node.** 28 nm.
- **Anchor (core).** core_mm2 `v('T1-OD-01')`; n_pub `s('T1-OD-01', 'scope', 'whole single-core processor', 1)`; its own share: own_memory_share.
- **Organization.** 256 x 256, 3 weight bits per cell, 4 stored bits per cell, 128 bits per neuron; term weights {'bits': 1.0}. Citations: rows, cols: T1-OD-01 scope (256 neurons, 64k synapses); stored_bits_per_cell: T1-OD-01 scope (64k 4-bit synapses); bits_per_neuron: frenkel2018a p03.raw.txt:266 'in 4kB and 32kB high-density single-port foundry SRAMs' / 256 neurons; weight_bits_per_cell: CO profiles/odin.json validity.geometry.cell_bits.
- **Dropped.** area_global_fixed 0.

Every figure is cited by its evidence id (`area_rule/data/evidence/published_figures.json`); none is typed in this description.
