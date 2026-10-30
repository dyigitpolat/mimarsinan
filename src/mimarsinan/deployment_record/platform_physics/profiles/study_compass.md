# COMPASS-geometry tile of the co-design study: the study's own chip as a profile

The co-design study (its estimator `RA/06_build/codesign/cd_preview.py`, `RA/06_build/x9/x9_0_v3.py`,
`RA/06_build/codesign/v41/cd41_chip.py`, `v42/cd42_common.py`, `v43/cd43_common.py`; its record
`RA/06_parts/codesign.md` and `RA/06_parts/constants.md`; its tile model, the nangate45 physical-design probe of
`RA/06_build/audit_v4/pd.md` with `RA/06_build/pd_probe/pd_table.json`) prices its designs on three tiles. This
profile is the COMPASS-geometry tile: the geometry of park2025compass Table I p.4 (256 x 256 one-bit arrays; chips
S, M and L of 144, 256 and 576 arrays) declared with exact digital accumulation, so one tile holds 256 rows by 64
four-bit weights (four one-bit columns per weight): 65,536 bits, 8 KiB, 16,384 MACs per activation
(constants.md sec. 2). COMPASS itself is analog and ADC-read; the study borrows its geometry only.

The study prices this tile on its SRAM-DDR technology row: jiang2023a's digital SRAM-CIM MAC energy (16 nm silicon,
full text) with a DDR3/4 backing store (jouppi2021ten Table 2), and its area from the probe's routed G3 tile (256 x
64 x 4 at nangate45, fakeram45 macro views). Each constant below is the study's own value mapped one to one onto
this vocabulary, with the study's source and class. The target bundle `study_compass` declares the core (256 axons
by 64 neurons of 4-bit weights, 576 tiles: the study's budget chip A_L), the lanes and the permissions.

## Validity

`technology_node_nm` 45 and `supply_v` 1.1 name the probe's nangate45 typical corner, the node of the study's tile
area (`cd_preview.TILE_MM2`): the lane block `ng45_os_lane` (45 nm, 1.1 V) then enters unscaled, which is how the
study's capability-on tile-area factor s_AU reads that same block (codesign.md sec. 2.2, D136; arith_toggle.md sec.
6). The MAC energy is jiang2023a's at 16 nm, as the study prices it: the study mixes the two nodes, and this profile
keeps its mix. `measurement_kind` mixed: silicon (MAC energy), simulation (tile area), estimates (the rows the study
does not price). The structured geometry is the toolchain's core of 256 x 64 four-bit slots; no aggregate constant
is declared, so it bounds nothing priced.

## Rows

| constant | value | evidence | basis |
|---|---|---|---|
| `e_mac` | 0.158730 pJ per MAC | derived | `cd_preview.A_TILE_J['SRAM-DDR']` = 2.6006 nJ per activation of a 65,536-bit tile over its 16,384 MACs: jiang2023a p.4, 12.6 TOPS/W non-weight-stationary 8b-W x 4b-input MAC, normalized to 4b-W x 8b-A by operand bits (constants.md sec. 4) |
| `t_cycle` | 5.333 - 87.912 ns, nominal 87.912 ns | derived | one latency step is one tile activation: T_ACT = 8 cycles (8-bit activations bit-serial, `cd41_chip.T_ACT`) at 91 MHz (jiang2023a p.4, the counted context) and 1.5 GHz (fujiwara2024344, abstract) |
| `area_per_cell` | 4.149200 um2 per 4-bit slot | derived | the probe's macro area per bit, 1.0373 um2 (`pd_table.json` g3 `macro_um2_per_bit`), times 4 bits |
| `area_per_neuron_logic` | 741.798 um2 per column | derived | the routed tile's 47,475.1 um2 of standard cells (MAC, adder tree, accumulators, input register, write and read ports) over 64 columns |
| `area_per_cell_per_weight_bit` | 0 | derived | inside `area_per_cell`: a slot's area is its 4 bits whatever the program's width |
| `area_per_row_driver` | 0 | derived | inside `area_per_neuron_logic`: the input register is standard cells of the tile |
| `e_core_init` | 0 | derived | inside `e_mac`: an accumulator clear is part of a tile activation |
| `e_dma_per_byte` | 162.5 pJ, high 325 pJ | derived | DDR3/4 I/O 1300 pJ per 64 bits (jouppi2021ten Table 2 p.3): the study's write and carry cost per byte; high end x 2, the write-and-carry factor f at its admissibility end (codesign.md sec. 3.3) |
| `e_readout_per_byte` | 162.5 pJ, high 325 pJ | derived | the same DDR3/4 I/O crossing for the read-back of a carried value: the study counts a carried value in both directions, written out and read back (`x9_0_v3._front_mincarry`, 2 x the live elements at each cut; one crossing per direction at 8 bits), so the read-back is its carry term as the re-injection on `e_dma_per_byte` is; high end x 2, the factor f |
| `e_write_per_cell` | 38 fJ, high 342 fJ | derived | the array write per byte, 76 fJ (the study's end of jiang2023a's 76 to 342 fJ bracket, constants.md sec. 5) per half-byte slot; high end the bracket's top at f = 2 |
| `e_program_per_byte` | 0.076 pJ, high 0.684 pJ | derived | the same array write per byte, for a program load's commit |
| `write_verify_iterations` | 1 | derived | a digital SRAM write has no verify loop |
| `t_program_per_byte` | 0.0278 - 0.3434 ns, nominal 0.3434 ns | derived | the study's write port, one 256-bit tile row per cycle (`cd41_chip.PORT_DEFAULT`, pd.md sec. 3), at 91 MHz; at 1.5 GHz the two DDR4 channels (36 GB/s, wang2020shuhai Table V p.7) bind |
| `pulses_per_write` | 1 | derived | one SRAM write is one write cycle |
| `conductance_levels` | 16 | derived | a 4-bit weight slot |
| `write_sigma`, `read_sigma` | 0 | derived | exact digital storage and accumulation (constants.md sec. 3, 4) |
| `membrane_bits` | 24 | derived | the probe tile's accumulators, A = 24 (pd.md sec. 3) |
| `host_compute_rate` | 1 | derived | identity on a measured host wall (the host is not part of the study's chip) |

## Not priced by the study

The study's energy is compute, writes and carry; its area is tiles (and port area beyond the default 256-bit port,
zero here); its time is compute and writes. Every other quantity the toolchain prices carries a sourced value with
its class and band, never zero, and its note starts `[not priced by the study]`:

| constant | value | evidence | source |
|---|---|---|---|
| `p_static_per_core` | 1 / 50 / 2,963.86 uW | estimated | low and nominal the generic 22 nm bundle's band (TrueNorth 10.25 uW and Loihi 815 uW per core bracket it); high the study's own routed G3 tile's leakage, 2.96386 mW (OpenSTA at default activity, nangate45 typical, `pd_table.json` g3 `leak_mw`) |
| `p_static_global` | 0.5 / 5 / 50 mW | estimated | the generic bundle's band, its low end raised from 0 to an order below its nominal |
| `e_noc_value_per_byte_hop` | 0.5 / 1 / 4 pJ | estimated | the generic bundle's band (Horowitz-scale wire plus router) |
| `e_inter_tile_hop`, `e_intra_tile_packet`, `t_hop` | the generic bundle's bands | estimated | TrueNorth and Loihi published points (spike traffic only) |
| `e_sync_barrier`, `t_sync_barrier` | 0.1 / 1 / 10 nJ; 100 / 465 / 2,000 ns | estimated | the generic bundle's bands; the study declares an on-chip pass sequencer at t_sync 0, without a source |
| `area_per_router` | 22,308.6 / 22,308.6 / 150,293 um2 | derived | [U-A] the measured router component, per NoC tile (section Chip area from components) |
| `area_global_fixed` | 20.35 / 25.72 / 80.88 mm2 | derived | [U-A] the sum of the chip_components (section Chip area from components); the study's A_L counts tiles only |
| `e_core_program` | 64 / 512 / 4,096 pJ | estimated | the generic bundle's band (a core-state sweep); the study's tile write is its bytes only |
| `bytes_per_connectivity_entry` | 4 / 8 / 16 B | estimated | the generic bundle's band; the study's payload is the tile bytes only |
| `endurance_cycles` | 1e16 | published | ielmini2018inmemory p.7 (embedded SRAM); the study's SRAM rows carry no endurance limit |
| `p_carry_buffer_static_per_byte` | 0.122 / 6.10 / 361.8 nW per byte | estimated | `p_static_per_core` over the core's 65,536 bit cells, x 8: the whole core leakage on its bitcells, an upper bound (section "On-chip carry buffer") |
| `area_carry_buffer_per_byte` | 8.2984 um2 per byte | derived | the chip's own cells per bit (1.0373 um2, `pd_table.json` g3 `macro_um2_per_bit`), x 8; bitcells only |


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
- **Static power per byte** (`p_carry_buffer_static_per_byte`, 0.122 / 6.10 / 361.8 nW, estimated): the chip's declared leakage
  per core (`p_static_per_core`, 1 / 50 / 2,963.86 uW) over the 65,536 bit cells of its declared core, x 8. The whole core
  leakage is charged to its bitcells, so this is an upper bound on a buffer of the chip's own cells. It is priced on
  the declared capacity, filled or not.
- **Area per byte** (`area_carry_buffer_per_byte`, 8.2984 um2, derived): the chip's own cells per bit (a cell is a
  4-bit weight slot, `area_per_cell`, plus 4 x `area_per_cell_per_weight_bit` = 0, over its 4 bits: the G3 tile's
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
(the probe's instance area, 0.115456 mm2 here, rounded 0.1155) already holds its SRAM macros (`area_per_cell`),
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

On one resident design the toolchain's compute term equals the study's n_act x A_tile where every layer fills its
tiles; otherwise it is the study's times the tile occupancy, because the study charges every tile activation in
full (its census counts whole tiles). The toolchain adds the lane block's energy (the study prices none) and the
value NoC; its area adds one router per NoC tile and the chip-level area of its declared components (section Chip area from components), and prices one lane per core slot
of its floorplan, where the study multiplies every tile by s_AU = 1.130 (the block's high end is 13.0% of a G3 tile:
15,000.1 um2 against the study's 15,015 um2). The tile part is the probe's instance area (0.115456 mm2), which the
study rounds to 0.1155 mm2. Carry: both count two crossings per carried value, written out (`e_readout_per_byte`) and read back in (`e_dma_per_byte`);
the study at its 8-bit activations (one byte per element per direction) and at its residency plan's cuts only (a resident
design carries nothing), the toolchain at the program's value grid (`activation_bits`) and at every pass boundary of its
schedule. The pinned
reconciliation is `tests/unit/deployment_record/introspection/test_study_cost_cell_ue4.py`.
