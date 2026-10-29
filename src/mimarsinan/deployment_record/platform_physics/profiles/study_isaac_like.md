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
| `area_per_router`, `area_per_tile_fixed` | the generic bundle's bands | estimated | per NoC tile |
| `area_global_fixed` | 2 / 10 / 40 mm2 | estimated | the generic bundle's band; the study's A_L counts tiles only |
| `e_core_program` | 64 / 512 / 4,096 pJ | estimated | the generic bundle's band |
| `bytes_per_connectivity_entry` | 4 / 8 / 16 B | estimated | the generic bundle's band |
| `endurance_cycles` | 1e16 | published | ielmini2018inmemory p.7 |

## Reconciliation with isaac_like

The shipped `isaac_like` profile is ISAAC itself (shafiee2016isaac): a 32 nm analog ReRAM design-space point,
bit-sliced crossbars read by 8-bit ADCs (`conversion_model` bit_sliced_crossbar, a converter table at 8 bits), its
constants derived from Table I powers at the 100 ns cycle. The study's ISAAC-like tile is a digital SRAM tile with
ISAAC's geometry (conversion model digital: no converter, a fact of exact accumulation, not an absence). The two
profiles describe different chips; `isaac_like` stays as it ships. Every constant that differs, and why:

| constant | isaac_like | study_isaac_like | why |
|---|---|---|---|
| `e_mac` | 0.0018311 pJ per cell-MAC | 0.158730 pJ per 4b x 8b MAC | ISAAC's array power per 1-bit-input x 2-bit-cell operation, against the study's jiang2023a digital MAC per weight slot |
| `t_cycle` | 100 - 200 ns | 5.333 - 87.912 ns | ISAAC's 100 ns crossbar-read stage against the study's 8 bit-serial cycles per activation at 91 MHz or 1.5 GHz |
| `t_array_read` | 100 - 200 ns | not declared | ISAAC's crossbar read; the study's activation time is `t_cycle` |
| `area_per_cell` | 0.0015259 um2 per 2-bit cell | 4.149207 um2 per 4-bit slot | a ReRAM cell (Table I) against the probe's SRAM macro per weight slot |
| `area_per_cell_per_weight_bit` | not declared | 0 | the study's slot area is per slot |
| `area_per_neuron_logic` | not declared | 371.245 - 668.370 um2 | the study's tile logic (MAC, adder tree, accumulators); ISAAC prices periphery per ADC and per tile |
| `area_per_row_driver` | 0.166 um2 per 1-bit DAC | 0 | ISAAC's DAC; the study's row drive is inside its tile logic |
| `area_per_adc` | 1,200 um2 | not declared | ISAAC's 8-bit ADC; the study's tile has no converter |
| `adc_sharing_factor` | 128 columns | not declared | ISAAC's converter sharing; no converter |
| `e_adc_conversion` | 1.5625 - 1.667 pJ | not declared | no converter |
| `t_adc_conversion` | 0.781 - 0.833 ns | not declared | no converter |
| `e_row_drive` | 0.3906 pJ | not declared | ISAAC's DAC energy; inside the study's MAC energy |
| `area_per_router` | 151,000 um2 | 1,000 - 15,000 um2 | ISAAC's c-mesh router (Table I) against the generic band (not priced by the study) |
| `area_per_tile_fixed` | 177,250 um2 | 2,000 - 30,000 um2 | ISAAC's tile (eDRAM, bus, sigmoid, pooling) against the generic band (not priced by the study) |
| `area_global_fixed` | 22.88 mm2 | 2 - 40 mm2 | ISAAC's HyperTransport links against the generic band (not priced by the study) |
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

Beside the constants: the node (32 nm against the probe's 45 nm), the supply (unstated against 1.1 V), the
measurement kind (simulation against mixed), the conversion model and converter table (ISAAC's 8-bit ADC against
none), and the geometry (128 x 128 two-bit cells as 128 axons by 128 neurons, against 128 x 64 four-bit slots).

## Terms that differ from the study's estimator

As `study_compass.md`: the compute term is the study's times the tile occupancy; the lane block's energy and the
value NoC are added; the area adds routers, NoC-tile overhead and the fixed global area, and the lane is one block
per core slot (13,047.9 um2 nominal, 15,000.1 um2 high) where the study scales this tile by s_AU = 1.130 (13.0% of
0.07677 mm2, 9,980 um2): the toolchain's lane is about 1.5x the study's on this tile. The tile part is the probe's
instance area 0.076766 mm2, which the study rounds to 0.07677 mm2.
