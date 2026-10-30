# ISAAC-geometry tile of the co-design study: the study's own chip as a profile

The co-design study (its estimator `RA/06_build/codesign/cd_preview.py`, `RA/06_build/x9/x9_0_v3.py`,
`RA/06_build/codesign/v41/cd41_chip.py`, `v42/cd42_common.py`, `v43/cd43_common.py`; its record
`RA/06_parts/codesign.md` and `RA/06_parts/constants.md`; its tile model, the nangate45 physical-design probe of
`RA/06_build/audit_v4/pd.md` with `RA/06_build/pd_probe/pd_table.json`) prices its designs on three tiles. This
profile is the ISAAC-like tile: ISAAC's 128 x 128 two-bit cell geometry (constants.md sec. 2: "ISAAC-like geometry
declared with exact accumulation", ISAAC lends the geometry only) as a digital SRAM-CIM tile, so one tile holds 128
rows by 64 four-bit weights (two two-bit cells per weight): 32,768 bits, 4 KiB, 8,192 MACs per activation.

The study prices this tile on its SRAM-DDR technology row: jiang2023a's digital SRAM-CIM MAC energy at the same
energy per MAC as its 65,536-bit tiles (`A_TILE_J` x 32,768 / 65,536 per activation), a DDR3/4 backing store, and
its area from the probe's routed G6 tile (128 x 64 x 4 at nangate45, fakeram45 macro views, 0.07677 mm2), with its
equal-area-per-bit model (0.5 x 0.1155 = 0.05775 mm2) as the band's low end (codesign.md sec. 3.3, D111; the
declared read's fakeram model is the nominal). The target bundle `study_isaac_like` declares the core (128 axons by
64 neurons of 4-bit weights, 576 tiles), the lanes and the permissions.

## Validity

`technology_node_nm` 45 and `supply_v` 1.1 name the probe's nangate45 typical corner, the node of the study's tile
area: the lane block `ng45_os_lane` (45 nm, 1.1 V) enters unscaled, as the study's s_AU reads it. The MAC energy is
jiang2023a's at 16 nm, as the study prices it. `measurement_kind` mixed. The structured geometry is the toolchain's
core of 128 x 64 four-bit slots; no aggregate constant is declared.

## Rows

| constant | value | evidence | basis |
|---|---|---|---|
| `e_mac` | 0.158730 pJ per MAC | derived | `A_TILE_J['SRAM-DDR']` x 32,768 / 65,536 per activation over 8,192 MACs: jiang2023a p.4, 12.6 TOPS/W non-WS, normalized to 4b-W x 8b-A (constants.md sec. 4) |
| `t_cycle` | 5.333 - 87.912 ns, nominal 87.912 ns | derived | T_ACT = 8 cycles per tile activation at 91 MHz (jiang2023a p.4) and 1.5 GHz (fujiwara2024344) |
| `area_per_cell` | 4.149207 um2 per 4-bit slot | derived | the G6 probe's macro area per bit, 1.0373016 um2 (`pd_table.json` g6), times 4 bits |
| `area_per_neuron_logic` | 371.245 - 668.370 um2 per column, nominal 668.370 | derived | the G6 tile's 42,775.7 um2 of standard cells over 64 columns; low end the equal-area-per-bit tile (0.05775 mm2) with its macro bits kept |
| `area_per_cell_per_weight_bit` | 0 | derived | inside `area_per_cell` |
| `area_per_row_driver` | 0 | derived | inside `area_per_neuron_logic` |
| `e_core_init` | 0 | derived | inside `e_mac` |
| `e_dma_per_byte` | 162.5 pJ, high 325 pJ | derived | DDR3/4 I/O 1300 pJ per 64 bits (jouppi2021ten Table 2 p.3); high end the write-and-carry factor f = 2 |
| `e_readout_per_byte` | 162.5 pJ, high 325 pJ | derived | the same DDR3/4 I/O crossing for the read-back direction of a carried value, which the study's carry counts (both directions, as `study_compass.md`); high end f = 2 |
| `e_write_per_cell` | 38 fJ, high 342 fJ | derived | the array write per byte, 76 fJ (constants.md sec. 5), per half-byte slot; high end at f = 2 |
| `e_program_per_byte` | 0.076 pJ, high 0.684 pJ | derived | the same array write per byte at a program load |
| `write_verify_iterations` | 1 | derived | a digital SRAM write has no verify loop |
| `t_program_per_byte` | 0.0278 - 0.3434 ns, nominal 0.3434 ns | derived | one 256-bit tile row per cycle (128 cells x 2 bits) at 91 MHz; at 1.5 GHz the DDR4 store binds |
| `pulses_per_write` | 1 | derived | one SRAM write is one write cycle |
| `conductance_levels` | 16 | derived | a 4-bit weight slot |
| `write_sigma`, `read_sigma` | 0 | derived | exact digital storage and accumulation |
| `membrane_bits` | 24 | derived | the probe tile's 24-bit accumulators |
| `host_compute_rate` | 1 | derived | identity on a measured host wall |

## Not priced by the study

As `study_compass.md`, with the G6 tile's leakage at the static band's high end:

| constant | value | evidence | source |
|---|---|---|---|
| `p_static_per_core` | 1 / 50 / 1,846.22 uW | estimated | generic band low and nominal; high the routed G6 tile's leakage, 1.84622 mW (`pd_table.json` g6 `leak_mw`) |
| `p_static_global` | 0.5 / 5 / 50 mW | estimated | the generic bundle's band, low end raised from 0 |
| `e_noc_value_per_byte_hop` | 0.5 / 1 / 4 pJ | estimated | the generic bundle's band |
| `e_inter_tile_hop`, `e_intra_tile_packet`, `t_hop` | the generic bundle's bands | estimated | TrueNorth and Loihi published points |
| `e_sync_barrier`, `t_sync_barrier` | the generic bundle's bands | estimated | the study's on-chip pass sequencer at t_sync 0 has no source |
| `area_per_router` | 22,308.6 / 22,308.6 / 150,293 um2 | derived | [U-A] the measured router component, per NoC tile (section Chip area from components) |
| `area_global_fixed` | 20.35 / 25.72 / 80.88 mm2 | derived | [U-A] the sum of the chip_components (section Chip area from components); the study's A_L counts tiles only |
| `e_core_program` | 64 / 512 / 4,096 pJ | estimated | the generic bundle's band |
| `bytes_per_connectivity_entry` | 4 / 8 / 16 B | estimated | the generic bundle's band |
| `endurance_cycles` | 1e16 | published | ielmini2018inmemory p.7 |
| `p_carry_buffer_static_per_byte` | 0.244 / 12.2 / 450.7 nW per byte | estimated | `p_static_per_core` over the core's 32,768 bit cells, x 8: the whole core leakage on its bitcells, an upper bound (section "On-chip carry buffer") |
| `area_carry_buffer_per_byte` | 8.2984 um2 per byte | derived | the chip's own cells per bit (1.0373 um2, `pd_table.json` g6 `macro_um2_per_bit`), x 8; bitcells only |

## Reconciliation with isaac_like

The shipped `isaac_like` profile is ISAAC itself (shafiee2016isaac): a 32 nm analog ReRAM design-space point,
bit-sliced crossbars read by 8-bit ADCs (`conversion_model` bit_sliced_crossbar, a converter table at 8 bits), its
constants derived from Table I powers at the 100 ns cycle. The study's ISAAC-like tile is a digital SRAM tile with
ISAAC's geometry (conversion model digital: no converter, a fact of exact accumulation, not an absence). The two
profiles describe different chips; `isaac_like` stays as it ships, its area reconciled to Table I by U-A (its IMA periphery per row and column, a quarter router per tile, its links as chip components). Every constant that differs, and why:

| constant | isaac_like | study_isaac_like | why |
|---|---|---|---|
| `e_mac` | 0.0018311 pJ per cell-MAC | 0.158730 pJ per 4b x 8b MAC | ISAAC's array power per 1-bit-input x 2-bit-cell operation, against the study's jiang2023a digital MAC per weight slot |
| `t_cycle` | 100 - 200 ns | 5.333 - 87.912 ns | ISAAC's 100 ns crossbar-read stage against the study's 8 bit-serial cycles per activation at 91 MHz or 1.5 GHz |
| `t_array_read` | 100 - 200 ns | not declared | ISAAC's crossbar read; the study's activation time is `t_cycle` |
| `area_per_cell` | 0.0015259 um2 per 2-bit cell | 4.149207 um2 per 4-bit slot | a ReRAM cell (Table I) against the probe's SRAM macro per weight slot |
| `area_per_cell_per_weight_bit` | not declared | 0 | the study's slot area is per slot |
| `area_per_neuron_logic` | 1.025 um2 per column | 371.245 - 668.370 um2 | [U-A] ISAAC's IMA column periphery (S+H, S+A, OR; Table I) against the study's tile logic (MAC, adder tree, accumulators) |
| `area_per_row_driver` | 2.217 um2 per row | 0 | [U-A] ISAAC's 1-bit DAC and input-register share (Table I); the study's row drive is inside its tile logic |
| `area_per_adc` | 1,200 um2 | not declared | ISAAC's 8-bit ADC; the study's tile has no converter |
| `adc_sharing_factor` | 128 columns | not declared | ISAAC's converter sharing; no converter |
| `e_adc_conversion` | 1.5625 - 1.667 pJ | not declared | no converter |
| `t_adc_conversion` | 0.781 - 0.833 ns | not declared | no converter |
| `e_row_drive` | 0.3906 pJ | not declared | ISAAC's DAC energy; inside the study's MAC energy |
| `area_per_router` | 37,750 um2 per tile | 22,308.6 - 150,293 um2 | [U-A] a quarter of ISAAC's c-mesh router per tile (Table I) against the measured router component (not priced by the study) |
| `area_per_tile_fixed` | 177,250 um2 | 0 | ISAAC's tile (eDRAM, bus, sigmoid, pooling) against [U-A] the study tile's own control and ports, inside its logic |
| `area_global_fixed` | 22.88 mm2 | 20.35 - 80.88 mm2 | [U-A] ISAAC's four HyperTransport links against the study chip's components (edge host link, its own DDR4 store's interfaces, controller, PLL, pads; not priced by the study) |
| `p_static_global` | 10,400 mW | 0.5 - 50 mW | ISAAC's constant HyperTransport power against the generic band (not priced by the study) |
| `p_static_per_core` | not declared | 1 - 1,846.22 uW | the G6 tile's leakage at the high end (not priced by the study) |
| `e_dma_per_byte` | 2.0215 pJ | 162.5 - 325 pJ | ISAAC's on-tile eDRAM buffer byte against the study's DDR3/4 backing-store I/O |
| `e_readout_per_byte` | not declared | 162.5 - 325 pJ | the read-back crossing of a carried value, which the study's carry counts (both directions) |
| `e_program_per_byte` | 237.7 - 2,377 pJ | 0.076 - 0.684 pJ | a ReRAM program-verify commit against the study's SRAM array write |
| `e_write_per_cell` | not declared | 38 - 342 fJ | the study's array write per slot; ISAAC prices its commit per byte |
| `t_program_per_byte` | 5.82 - 1,454.55 ns | 0.0278 - 0.3434 ns | a donor-scaled configuration port against the study's 256-bit row per cycle |
| `t_write_per_unit` | 2 - 11 us | not declared | ReRAM write-verify per cell; an SRAM write is priced per byte at the port |
| `write_verify_iterations` | 8.52 | 1 | NeuRRAM's write-verify statistics against a digital write |
| `pulses_per_write` | 8.52 | 1 | the same |
| `endurance_cycles` | 1e10 - 1e11 | 1e16 | a resistive cell against embedded SRAM |
| `e_core_program` | 1.035 - 4.14 nJ | 64 - 4,096 pJ | ISAAC's tile eDRAM cycle against the generic band (not priced by the study) |
| `e_core_init` | not declared | 0 | inside the study's MAC energy |
| `e_intra_tile_packet` | 0.6836 pJ per byte | 0.1 - 1.5 pJ | ISAAC's bus against the generic band |
| `e_inter_tile_hop` | not declared | 0.5 - 4 pJ | the generic band |
| `t_hop` | not declared | 1 - 8 ns | the generic band |
| `e_noc_value_per_byte_hop` | 1.09 - 1.31 pJ | 0.5 - 4 pJ | ISAAC's router power per byte-hop against the generic band (not priced by the study) |
| `e_sync_barrier` | not declared | 0.1 - 10 nJ | the generic band |
| `t_sync_barrier` | not declared | 100 - 2,000 ns | the generic band |
| `bytes_per_connectivity_entry` | not declared | 4 - 16 B | the generic band |
| `conductance_levels` | 4 | 16 | a 2-bit ReRAM cell against a 4-bit weight slot |
| `membrane_bits` | 16 | 24 | ISAAC's 16-bit fixed point against the probe tile's 24-bit accumulators |
| `write_sigma` | not declared | 0 | exact digital storage |
| `read_sigma` | not declared | 0 | exact digital accumulation |
| `host_compute_rate` | 1 | 1 | the same identity, its note naming that the host is not the study's chip |
| `p_carry_buffer_static_per_byte` | not declared | 0.244 / 12.2 / 450.7 nW per byte | [U-G] the study chip's carry buffer: its own leakage over its 32,768 bit cells (not priced by the study) |
| `area_carry_buffer_per_byte` | not declared | 8.2984 um2 per byte | [U-G] the study chip's carry buffer: its own cells per bit, bitcells only (not priced by the study) |

Beside the constants: the node (32 nm against the probe's 45 nm), the supply (unstated against 1.1 V), the
measurement kind (simulation against mixed), the conversion model and converter table (ISAAC's 8-bit ADC against
none), the carry buffer table (none against U-G's three 45 nm SRAM rows), and the geometry (128 x 128 two-bit cells as 128 axons by 128 neurons, against 128 x 64 four-bit slots).


## On-chip carry buffer (U-G)

The levers scout on the study chips (`RA/11_wave2/levers_study/scout.md` sections 2, 6, 8 and 9) found the carry
the strongest lever: every pass boundary of a one-copy design crosses the DDR store. This chip states an on-chip
carry buffer, U-F4's declared component: a run declares its capacity (`carry_buffer_bytes`), a wavefront group's own
values then stay in it (written once, read once), and a group whose live values exceed the capacity is refused by
name. The study's estimator holds no on-chip buffer, so every buffer constant is `[not priced by the study]`.

- **Access energy over capacity** (the profile's `carry_buffer` table, rows of `e_carry_buffer_per_byte`): an SRAM's
  access energy depends on its size, so each row is the 45 nm figure (the probe's node, `technology_node_nm` 45) of an
  SRAM of that size, jouppi2021ten Table 2 p.3 ("Memory is pJ per 64-bit access"), over 8 bytes, a point value:

  | capacity | Table 2 row (45 nm) | e_carry_buffer_per_byte | evidence |
  |---|---|---|---|
  | 8,192 B | 8 KB SRAM, 10 pJ per 64-bit access | 1.25 pJ per byte | derived |
  | 32,768 B | 32 KB SRAM, 20 pJ | 2.5 pJ per byte | derived |
  | 1,048,576 B | 1 MB SRAM, 100 pJ (footnote 1: a single bank SRAM) | 12.5 pJ per byte | derived |

  The source's KB and MB are read as binary capacities. A run declaring a capacity the table does not state is
  refused the buffer energy by name (the pricer never interpolates). The declared read does not scale these rows.
- **Static power per byte** (`p_carry_buffer_static_per_byte`, 0.244 / 12.2 / 450.7 nW, estimated): the chip's declared leakage
  per core (`p_static_per_core`, 1 / 50 / 1,846.22 uW) over the 32,768 bit cells of its declared core, x 8. The whole core
  leakage is charged to its bitcells, so this is an upper bound on a buffer of the chip's own cells. It is priced on
  the declared capacity, filled or not.
- **Area per byte** (`area_carry_buffer_per_byte`, 8.2984 um2, derived): the chip's own cells per bit (a cell is a
  4-bit weight slot, `area_per_cell`, plus 4 x `area_per_cell_per_weight_bit` = 0, over its 4 bits: the G6 tile's
  macro area per bit, `pd_table.json` `macro_um2_per_bit`), x 8. Bitcells only: the buffer's decoder, sense-amplifier
  and driver periphery is not priced.

The capacity a study reading declares follows a size rule in its driver (`buffer_size_rule` of
`RA/12_tcad_readiness/e2e/w48/w48_pipelined.py`): the smallest power of two holding the largest live peak of the
priced members.
The pins are `tests/unit/deployment_record/platform_physics/test_study_chip_carry_buffer_ug.py` and
`test_carry_buffer_table_ug.py`.

## Chip area from components (U-A)

[U-A, owner instruction of 2026-10-28] The chip area is modeled from declared components, each a sourced area at its
own node with a declared node-scaling rule (`platform_physics/area_components`; rules `logic`, the node ratio
squared, and `io_phy`, unscaled nominal up to the logic rule, for PHYs, pads and analog blocks). The study's tile
(the probe's instance area, 0.076766 mm2 here, rounded 0.07677) already holds its SRAM macros (`area_per_cell`),
its MAC, adder tree, accumulators, input register and its weight write, activation write and accumulator read
ports, and its own pass control (`area_per_neuron_logic`, pd.md sec. 3); the lane is the `ng45_os_lane` block.
Beside them the toolchain prices, once each and named:

| block | per | value (45 nm) | source |
|---|---|---|---|
| router (`area_per_router`) | NoC tile of the resolved floorplan | 22,308.6 um2, high 150,293 um2 | OpenPiton's dynamic_node, five ports of 64-bit flits, ORFS nangate45 instance area on the probe's image (MEASURED); high end Simba's per-PE router (shao2019simba Table 3) at the logic rule |
| NoC-tile block (`area_per_tile_fixed`) | NoC tile | 0 | inside `area_per_neuron_logic` (the tile's control, registers and ports) and `area_per_router` (its input buffers); the glue between a NoC tile's tiles and the router's local port is not priced |
| host link PHY | 4 lanes | 2.475 mm2 (2.06 to 7.46) | the TPUv1's PCIe Gen3 interface per lane (jouppi2017indatacenter Fig. 2, 3% of a die below 330 mm2), for the PCIe Gen3 x4 accelerator attach of U-J's edge host `jetson_orin_nano_8gb` |
| memory PHY | 2 channels | 19.8 mm2 (16.5 to 59.7) | the TPUv1's DRAM port per 64-bit channel (3% of the die each): the study's store is the chip's own two DDR4 channels at 36 GB/s (`cd41_chip.CTX`, wang2020shuhai Table V p.7, the accelerator's own DDR4) |
| global controller | 1 | 0.0294 mm2 (high 0.862) | an ibex RV32 core, ORFS nangate45 instance area (MEASURED); high end Simba's RISC-V partition at the logic rule; its memories not priced |
| clock generation | 1 | 0.111 mm2 | a 45 nm digital PLL (goel2010a, abstract) |
| pads and misc I/O | 1 | 3.3 mm2 (1.65 to 12.8) | the TPUv1's misc I/O block (1% of the die) |

`area_global_fixed` is the sum of the last five: 25.72 mm2 nominal (20.35 to 80.88 mm2), against the generic
band's 2 / 10 / 40 mm2 the profile carried before. The two DDR x64 interfaces are 77% of it at nominal: whether the
chip owns its DRAM is the reading's largest area choice, and the study's sources make the store the chip's.

## Terms that differ from the study's estimator

As `study_compass.md`: the compute term is the study's times the tile occupancy; the lane block's energy and the
value NoC are added; the area adds one router per NoC tile and the chip-level area of its declared components (as `study_compass.md`, section Chip area from components), and the lane is one block
per core slot (13,047.9 um2 nominal, 15,000.1 um2 high) where the study scales this tile by s_AU = 1.130 (13.0% of
0.07677 mm2, 9,980 um2): the toolchain's lane is about 1.5x the study's on this tile. The tile part is the probe's
instance area 0.076766 mm2, which the study rounds to 0.07677 mm2.
