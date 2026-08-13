# ISAAC physics profile — published per-unit constants

Target: **ISAAC-CE**, the design point whose parameters, power and area breakdowns are
the ones actually tabulated in the paper (Table I).

> **ISAAC is a SIMULATION/MODELLING study, not silicon.** No ISAAC chip was ever
> fabricated. Every number below is an analytical model output. See §3.1 for the model
> stack. Treat all "published" rows as *published modelled values*, never as measurements.

---

## 0. Sources read

| key | source | type | how read |
|---|---|---|---|
| `shafiee2016isaac` | A. Shafiee, A. Nag, N. Muralimanohar, R. Balasubramonian, J. P. Strachan, M. Hu, R. S. Williams, V. Srikumar, "ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars", *ISCA-43*, 2016, pp. 14–26 | peer-reviewed paper (analytical model + circuit-survey data) | full PDF, 12 pp, text via `pdftotext -layout`; **Table I additionally re-read as a 300 dpi page render** to confirm the column alignment of the transcription below |

**Version note (load-bearing).** The retrieved PDF carries a banner on page 1:

> "October 5th 2016: This version corrects some of the results for the ISAAC-PE and ISAAC-SE configurations."

All ISAAC-CE numbers used here are from this corrected version. One stale figure survives
the erratum (the 1707 GOPS/s·mm² bare-array claim) — see §3.2.

---

## 1. Constants table

Vocabulary keys are those of `src/mimarsinan/deployment_record/platform_physics/vocabulary.py`.
`evidence_kind ∈ published | datasheet | derived | estimated`.

### 1.0 How to read Table I (this is the key to every derived row)

Table I's `Power` and `Area` columns are **group totals for that component within one IMA
(or one tile)** — *not* per-instance values. This is not stated in the paper, so it is
proven here by four independent reconciliations, all of which the per-instance reading
fails:

| check | group-total reading | paper's own statement |
|---|---|---|
| Σ IMA rows × 12 | 24.08 mW × 12 = **288.96 mW** ; 0.013120 mm² × 12 = **0.15744 mm²** | "IMA Total … 289 mW … 0.157 mm²" ✓ |
| Σ tile rows (router ÷ 4) | 40.85 mW ; 0.214850 mm² | "Total … 40.9 mW … 0.215 mm²" ✓ |
| ADC share of tile power | 12 × 16 mW = 192 mW → 192/330 = **58.2 %** | "the ADCs account for 58 % of tile power" ✓ |
| ADC share of tile area | 12 × 0.0096 = 0.1152 mm² → **31.0 %** | "…and 31 % of tile area" ✓ |
| eDRAM + bus area | (0.083+0.090)/0.372 = **46.5 %** | "eDRAM buffer and the eDRAM-IMA bus together take up 47 % of tile area" ✓ |

Under the per-instance reading the IMA area sums to 0.174 mm², contradicting the printed
0.157 mm². **Group-total reading adopted throughout.** Consequently every per-instance
constant below is `derived` by dividing by the printed instance count.

### 1.1 Table I transcribed in full (verbatim)

> Header row: **"ISAAC Tile at 1.2 GHz, 0.37 mm²"**

| Component | Params | Spec | Power | Area (mm²) |
|---|---|---|---|---|
| eDRAM Buffer | size / num_banks / bus_width | 64KB / 2 / 256 b | 20.7 mW | 0.083 |
| eDRAM-to-IMA bus | num_wire | 384 | 7 mW | 0.090 |
| Router | flit size / num_port | 32 / 8 | 42 mW | 0.151 (shared by 4 tiles) |
| Sigmoid | number | 2 | 0.52 mW | 0.0006 |
| S+A | number | 1 | 0.05 mW | 0.00006 |
| MaxPool | number | 1 | 0.4 mW | 0.00024 |
| OR | size | 3 KB | 1.68 mW | 0.0032 |
| **Total** | | | **40.9 mW** | **0.215 mm²** |

> Section header: **"IMA properties (12 IMAs per tile)"**

| Component | Params | Spec | Power | Area (mm²) |
|---|---|---|---|---|
| ADC | resolution / frequency / number | 8 bits / 1.2 GSps / 8 | 16 mW | 0.0096 |
| DAC | resolution / number | 1 bit / 8 × 128 | 4 mW | 0.00017 |
| S+H | number | 8 × 128 | 10 uW | 0.00004 |
| Memristor array | number / size / bits per cell | 8 / 128 × 128 / 2 | 2.4 mW | 0.0002 |
| S+A | number | 4 | 0.2 mW | 0.00024 |
| IR | size | 2 KB | 1.24 mW | 0.0021 |
| OR | size | 256 B | 0.23 mW | 0.00077 |
| **IMA Total** | number | 12 | **289 mW** | **0.157 mm²** |
| **1 Tile Total** | | | **330 mW** | **0.372 mm²** |
| **168 Tile Total** | | | **55.4 W** | **62.5 mm²** |
| Hyper Tr | links/freq / link bw | 4/1.6GHz / 6.4 GB/s | 10.4 W | 22.88 |
| **Chip Total** | | | **65.8 W** | **85.4 mm²** |

> Section header: **"DaDianNao at 606 MHz scaled up to 32nm"** (baseline, transcribed for completeness)

| Component | Params | Spec | Power | Area (mm²) |
|---|---|---|---|---|
| eDRAM | size / num_banks | 36 MB / 4 per tile | 4.8 W | 33.22 |
| NFU | number | 16 | 4.9 W | 16.22 |
| Global Bus | width | 128 bit | 13 mW | 15.7 |
| **16 Tile Total** | | | **9.7 W** | **65.1 mm²** |
| Hyper Tr | links/freq / link bw | 4/1.6GHz / 6.4 GB/s | 10.4 W | 22.88 |
| **Chip Total** | | | **20.1 W** | **88 mm²** |

Caption: **"TABLE I — ISAAC PARAMETERS."**

### 1.2 AREA

| quantity | value | band | unit | evidence_kind | citation | quote / arithmetic |
|---|---|---|---|---|---|---|
| **Technology node** | **32** | — | nm | published | `shafiee2016isaac`, §VII "Energy and Area Models", p.7 | "We use CACTI 6.5 [45] at 32 nm to model energy and area for all buffers and on-chip interconnects." and "we use an 8-bit ADC at 32 nm that is optimized for area." |
| **Die area (chip total)** | **85.4** | — | mm² | published | `shafiee2016isaac`, Table I, "Chip Total" row, p.8 | Table I cell: "Chip Total … 85.4 mm²" |
| Die area, tile array only (168 tiles) | 62.5 | — | mm² | published | `shafiee2016isaac`, Table I, "168 Tile Total" row | Table I cell: "168 Tile Total … 62.5 mm²" |
| `area_global_fixed` (HyperTransport off-chip I/O) | **22.88** | — | mm² | published | `shafiee2016isaac`, Table I, "Hyper Tr" row | Table I cell: "Hyper Tr / links/freq 4/1.6GHz / link bw 6.4 GB/s … 10.4 W … 22.88" |
| Tile area (complete) | 0.372 | — | mm² | published | `shafiee2016isaac`, Table I, "1 Tile Total" | "1 Tile Total … 0.372 mm²". Table header rounds it: "ISAAC Tile at 1.2 GHz, 0.37 mm²" |
| `area_per_tile_fixed` (non-IMA, non-router tile area) | **177 250** | — | µm² | derived | Table I tile block | 0.215 mm² (tile "Total") − 0.151/4 mm² (router share) = 0.17725 mm² = **177 250 µm²**. Components: eDRAM buffer 0.083 + bus 0.090 + sigmoid 0.0006 + S+A 0.00006 + maxpool 0.00024 + OR 0.0032 |
| `area_per_router` | **151 000** | — | µm² | published | `shafiee2016isaac`, Table I, "Router" row | "Router … 0.151 (shared by 4 tiles)". ⚠ **One router serves 4 tiles**: charge 0.151 mm²/router, or **37 750 µm² per tile** if amortising |
| `area_per_adc` | **1200** | — | µm² | derived | Table I, ADC row (0.0096 mm², number 8) | 0.0096 mm² ÷ 8 ADCs = 0.0012 mm² = **1200 µm²** per 8-bit 1.2 GSps ADC at 32 nm |
| `area_per_row_driver` (1-bit DAC) | **0.166** | — | µm² | derived | Table I, DAC row (0.00017 mm², number 8 × 128) | 0.00017 mm² = 170 µm²; ÷ (8 arrays × 128 rows = 1024 DACs) = **0.1660 µm²** per DAC. Plausibility: paper calls a 1-bit DAC "a trivial circuit (an inverter)"; ~0.17 µm² is a minimum-size inverter at 32 nm ✓ |
| `area_per_cell` | **1.526 × 10⁻³** | — | µm² | derived | Table I, Memristor array row (0.0002 mm², number 8, size 128×128) | 0.0002 mm² ÷ 8 arrays = 25 µm²/array; ÷ (128×128 = 16 384 cells) = **1.5259 × 10⁻³ µm² = 1526 nm²**. Implied cell pitch √1526 nm² = **39.1 nm** (= 4F² at F = 19.5 nm). ⚠ **Only 1 significant figure in the source (0.0002)** — see §2 and §3.2 |
| Area per memristor array (128×128, 2 b/cell) | 25 | — | µm² | derived | as above | 0.0002 mm² ÷ 8 = **25 µm²** |
| Area per S+H cell | 0.0391 | — | µm² | derived | Table I, S+H row (0.00004 mm², number 8 × 128) | 0.00004 mm² = 40 µm² ÷ 1024 = **0.03906 µm²** |
| `area_per_neuron_logic` (⚠ mapped, see note) | **300** | — | µm² | derived | Table I, Sigmoid row (0.0006 mm², number 2) | 0.0006 mm² ÷ 2 = 0.0003 mm² = **300 µm²** per sigmoid unit. ⚠ **This is a shared, time-multiplexed activation unit (2 per tile serving 12 IMAs), NOT per-neuron logic.** ISAAC has no per-neuron soma. Mapping it onto `area_per_neuron_logic` (multiplicand `neurons_physical`) would be a category error — see §2 |
| Area per MaxPool unit | 240 | — | µm² | published | Table I, MaxPool row | "MaxPool / number 1 … 0.00024" = 240 µm² per tile |
| Area per tile S+A unit | 60 | — | µm² | published | Table I, tile S+A row | "S+A / number 1 … 0.00006" = 60 µm² |
| Area per IMA S+A unit | 60 | — | µm² | derived | Table I, IMA S+A row (0.00024 mm², number 4) | 0.00024 ÷ 4 = 0.00006 mm² = **60 µm²** (identical to the tile S+A, a consistency check ✓) |
| IMA area (complete) | 13 120 | — | µm² | derived | Table I IMA block | 0.157 mm² ÷ 12 IMAs = **0.013083 mm²**; component sum = **0.013120 mm²** (0.3 % rounding) |

### 1.3 ENERGY

All ISAAC energies are `derived`: **Table I publishes power, never energy.** Conversion uses
the paper's own cycle, `t_cycle = 100 ns` (§1.5), and the printed instance counts.

| quantity | value | band | unit | evidence_kind | citation | arithmetic |
|---|---|---|---|---|---|---|
| `e_mac` (one 1-bit-input × 2-bit-cell array MAC) | **1.83 × 10⁻³** | — | pJ (= **1.83 fJ**) | derived | Table I memristor-array row + §IV cycle | 2.4 mW × 100 ns = 240 pJ per IMA-cycle; an IMA holds 8 × 128 × 128 = **131 072** cells, all activated every cycle → 240 pJ ÷ 131 072 = **1.8311 fJ** per cell-MAC |
| `e_mac` expressed per **full 16 b × 16 b MAC** | **0.234** | — | pJ | derived | as above | one 16×16 MAC = 16 input cycles × 8 weight cells = 128 cell-MACs → 128 × 1.8311 fJ = **234.4 fJ = 0.2344 pJ** (array only; excludes ADC/DAC/periphery) |
| `e_adc_conversion` | **1.5625** | 1.5625 – 1.667 | pJ | derived | Table I ADC row + §V "The Read/ADC Pipeline" | **Primary (operational):** 16 mW × 100 ns = 1.6 nJ per IMA-cycle; conversions per IMA-cycle = 8 arrays × 128 columns = **1024** → **1.5625 pJ**. **Upper band (nameplate):** 2 mW/ADC ÷ 1.2 GSps (Table I "frequency 1.2 GSps") = **1.667 pJ**. The two agree because the text's rate is 1.28 GSps: "these analog values in the sample-and-holds are fed sequentially to a single 1.28 giga-samples-per-second (GSps) ADC unit" |
| `e_row_drive` (one 1-bit DAC row activation) | **0.391** | — | pJ | derived | Table I DAC row | 4 mW × 100 ns = 400 pJ per IMA-cycle ÷ 1024 rows = **0.3906 pJ** |
| Energy per S+H sample | 9.77 × 10⁻⁴ | — | pJ | derived | Table I S+H row | 10 µW × 100 ns = 1.0 pJ per IMA-cycle ÷ 1024 = **0.977 fJ**. Cross-check: per-S+H power 10 µW/1024 = **9.77 nW**, and the cited S+H is O'Halloran & Sarpeshkar ref [48], *"A 10-nW 12-bit Accurate Analog Storage Cell"* ✓ exact |
| `e_intra_tile_packet` / `e_dma_per_byte` (on-tile, eDRAM→IMA bus only) | **0.684** | — | pJ/byte | derived | Table I bus row + §VI | 7 mW × 100 ns = 700 pJ per tile-cycle; the bus is sized for "up to 1KB of data from eDRAM to IR … within a 100 ns stage" → 700 pJ ÷ 1024 B = **0.6836 pJ/B** |
| eDRAM read energy | 2.02 | — | pJ/byte | derived | Table I eDRAM-buffer row | 20.7 mW × 100 ns = 2.07 nJ per tile-cycle ÷ 1024 B = **2.0215 pJ/B** |
| eDRAM read **+** bus transport, combined | 2.71 | — | pJ/byte | derived | as above | (20.7 + 7) mW × 100 ns ÷ 1024 B = **2.7051 pJ/B** |
| `e_inter_tile_hop` (32-bit flit, one router traversal) | 4.375 | 4.375 – 5.25 | pJ/flit·hop | **estimated** | Table I router row | 42 mW ÷ (8 ports × 1.2 GHz) = **4.375 pJ**; at the paper's conservative link rate ("we conservatively assume a 32-bit link operating at 1 GHz") 42 mW ÷ (8 × 1 GHz) = **5.25 pJ**. ⚠ **The activity factor is my assumption, not the paper's** — Table I gives router *power* only, with no stated utilisation. Do not promote to `derived` |
| `e_dma_per_byte` (off-chip, HyperTransport) | 406 | — | pJ/byte | derived (⚠ static-dominated) | Table I Hyper Tr row | 10.4 W ÷ (4 links × 6.4 GB/s = 25.6 GB/s) = **406.2 pJ/B**. ⚠ The paper states HT power is **constant, not activity-proportional**: "The HT is a constant overhead of 10 W, representing half of DaDianNao chip power, but only 16 % of ISAAC chip power." Prefer modelling it as `p_static_global = 10.4 W` |
| Sigmoid energy per tile-cycle | 52 | — | pJ | derived | Table I sigmoid row | 0.52 mW × 100 ns = **52 pJ** per tile-cycle (2 units). Per unit-cycle = 26 pJ. **Per-neuron energy is UNSOURCED** — the paper never states sigmoid throughput per cycle (§2) |
| MaxPool energy per tile-cycle | 40 | — | pJ | derived | Table I maxpool row | 0.4 mW × 100 ns = **40 pJ** |

### 1.4 TIME

| quantity | value | band | unit | evidence_kind | citation | quote |
|---|---|---|---|---|---|---|
| `t_array_read` / `t_cycle` | **100** | 100 – 200 | ns | published | `shafiee2016isaac`, §IV "The ISAAC Pipeline", p.4 | "In our discussions, a cycle is the time required to perform one crossbar read operation, which for most of our analysis is 100 ns." Upper band from §VIII-A: "if crossbar latency is assumed to be 200 ns, throughput is reduced by 2×, but because many of the structures become simpler, CE is only reduced by 30 %." |
| `t_cycle` (restated as the pipeline stage) | 100 | — | ns | published | `shafiee2016isaac`, §VI, p.6 | "This operation is itself pipelined (shown in detail in Figure 4b), with the cycle time (100 ns) dictated by the slowest stage, which is the crossbar read." |
| `t_adc_conversion` | **0.781** | 0.781 – 0.833 | ns | derived | `shafiee2016isaac`, §V + Table I | Text rate: 1/1.28 GSps = **0.7813 ns**; Table I nameplate: 1/1.2 GSps = **0.8333 ns**. Text: "128 bitline currents are processed in 100 ns" → 100 ns/128 = 0.781 ns ✓ |
| Digital clock (tile) | **1.2** | — | GHz | published | `shafiee2016isaac`, Table I header | "ISAAC Tile at 1.2 GHz, 0.37 mm²" |
| Digital clock period | 0.833 | — | ns | derived | as above | 1/1.2 GHz = **0.8333 ns**. Note 100 ns ÷ 0.833 ns = 120 digital cycles per crossbar read |
| IMA occupancy per 16-bit dot product | 16 | — | cycles (= 1.6 µs) | published | `shafiee2016isaac`, §V "Input Voltages and DACs" | "The process continues until all 16 bits of the input have been handled in 16 cycles." |
| Full IMA→eDRAM pipeline depth (worked example) | 22 | — | cycles | published | `shafiee2016isaac`, Fig. 4b + §VI | Fig. 4b: eDRAM Rd+IR (cyc 1) → Xbar (2–17) → ADC (18) → S+A/OR wr (19–20) → V/sigmoid (21) → eDRAM Wr (22) |
| `t_hop` | — | — | ns | **UNSOURCED** | — | Router latency is never stated; only flit size (32 b), port count (8) and power (42 mW). See §2 |
| HyperTransport link frequency / bandwidth | 1.6 GHz / 6.4 GB/s per link | — | — | published | Table I | "Hyper Tr / links/freq 4/1.6GHz / link bw 6.4 GB/s" |
| Inter-tile link (c-mesh) assumption | 32 b @ 1 GHz | — | — | published | `shafiee2016isaac`, §VIII-A, p.9 | "we estimate that the inter-tile link bandwidth requirement never exceeds 3.2 GB/s. We therefore conservatively assume a 32-bit link operating at 1 GHz." |

### 1.5 STRUCTURE (the ADC-conversion model's inputs)

| quantity | value | evidence_kind | citation | quote |
|---|---|---|---|---|
| **Crossbar array size** | **128 × 128** (rows × cols) | published | `shafiee2016isaac`, §VIII-A p.9 + Table I | "The optimal design point has 8 128×128 arrays, 8 ADCs per IMA, and 12 IMAs per tile. We refer to this design as ISAAC-CE." Table I: "Memristor array / size / 128 × 128" |
| Effective columns incl. **unit column** | **129** | published | `shafiee2016isaac`, §V "Encoding to Reduce ADC Size", p.5 | "In terms of overheads, the columns per array has grown from 128 to 129 and one additional shift-and-add has been introduced." The unit column computes Σaᵢ for the flipped-weight and bias corrections |
| **Arrays per IMA** | **8** | published | §VIII-A p.9 / Table I | (same quote as array size); Table I "Memristor array / number / 8" |
| **ADCs per IMA** | **8** | published | §VIII-A p.9 / Table I | "8 ADCs per IMA"; Table I "ADC / number / 8" |
| **ADCs per array** | **1** | derived | 8 ADCs ÷ 8 arrays | Confirmed by §V: "these bitline currents are latched in 128 sample-and-hold circuits. In the next 100 ns cycle, these analog values in the sample-and-holds are fed sequentially to a single 1.28 giga-samples-per-second (GSps) ADC unit." |
| **`adc_sharing_factor`** | **128** columns per ADC | derived (from an explicit dataflow statement) | `shafiee2016isaac`, §V, p.5 | "For example, a 128×128 crossbar may produce 128 bitline currents every 100 ns (one cycle, which is the read latency for the crossbar array)… Thus, 128 bitline currents are processed in 100 ns, before the next set of bitline currents are latched in the sample-and-hold circuits." → all 128 columns share 1 ADC |
| **ADC resolution** | **8** bits | published | Table I; §VIII-A | "ADC / resolution / 8 bits". §VIII-A: "we first confirmed that a 9-bit ADC is never worth the power/area overhead" |
| ADC sample rate | 1.2 GSps (Table I) / 1.28 GSps (text) | published | Table I; §V | "ADC / frequency / 1.2 GSps"; "a single 1.28 giga-samples-per-second (GSps) ADC unit" |
| **DAC resolution (`v`)** | **1** bit | published | Table I; §VIII-A | "DAC / resolution / 1 bit". "Performing a similar analysis for DAC resolution v yields maximal CE for 1-bit DACs." §V: "Note that a 1-bit DAC is a trivial circuit (an inverter)." |
| DACs per IMA | **1024** (8 × 128) | published | Table I | "DAC / number / 8 × 128" — one DAC per row per array |
| **Cell precision (`w`)** | **2** bits/cell (4 levels) | published | Table I; §VIII-A | "bits per cell / 2". "We empirically estimated that the CE metric is maximized when using 2 bits per cell (w = 2)." |
| **Cells per weight** | **8** | published | `shafiee2016isaac`, §V "Synaptic Weights and ADCs", p.5 | "We therefore represent one 16-bit synaptic weight with 16/w w-bit cells located in the same row. For the rest of this discussion, we assume w = 2" → 16/2 = 8 |
| **Weight precision** | **16** bits fixed point | published | `shafiee2016isaac`, §V, p.4 | "We target 16-bit fixed-point arithmetic, partially because prior work has shown that 16-bit arithmetic is sufficient for this class of machine learning algorithms" |
| **Input precision** | **16** bits, streamed 1 bit/cycle | published | `shafiee2016isaac`, §V, p.4 | "we provide 16 voltage levels sequentially, where voltage level i is a 0/1 binary input representing bit i of the 16-bit input number." |
| **IMAs per tile** | **12** | published | §VIII-A p.9; Table I | "12 IMAs per tile"; Table I "IMA properties (12 IMAs per tile)" |
| **Tiles per chip** | **168** (14 × 12) | published | `shafiee2016isaac`, §VII p.7 + Table I | "our analysis in Table I shows that one ISAAC chip can accommodate 14×12 tiles"; Table I "168 Tile Total" |
| Arrays per chip | **16 128** | derived | 8 × 12 × 168 | 8 arrays/IMA × 12 IMAs/tile × 168 tiles = **16 128** |
| ADCs per tile / per chip | 96 / 16 128 | published / derived | §VIII-A p.10 | "the 96 ADCs in a tile account for 58 % of tile power" ✓ (= 12 × 8); chip = 168 × 96 = 16 128 |
| ADC-resolution law | `A = log(R)+v+w` if v>1 ∧ w>1; else `A = log(R)+v+w−1` | published | `shafiee2016isaac`, §V eqs. (1)–(2), p.5 | With R=128, v=1, w=2: A = 7+1+2−1 = 9; the flipped-weight encoding saves one more bit → **8-bit ADC**. Confirmed: "Without the encoding scheme, we would either need a 9-bit ADC or half as many rows per crossbar array." |
| Signed arithmetic | inputs 2's complement (last cycle shift-and-*subtract*); weights biased by 2¹⁵ | published | `shafiee2016isaac`, §V "Correctly Handling Signed Arithmetic" | "the result of the last dot-product operation undergoes a shift-and-subtract instead of a shift-and-add"; "A 16-bit fixed-point weight between −2¹⁵ and 2¹⁵−1 is represented by an unsigned 16-bit integer, and the conversion is performed by subtracting a bias of 2¹⁵." |
| eDRAM buffer per tile | 64 KB, 2 banks, 256 b bus | published | Table I; §VIII-A | "The size of the central eDRAM buffer in a node is set to 64 KB and the c-mesh flit width is set to 32 bits." |
| Interconnect topology | on-chip **concentrated mesh (c-mesh)**, statically scheduled | published | `shafiee2016isaac`, §III, p.3 + §VI | "an ISAAC chip is composed of a number of tiles (labeled T), connected with an on-chip concentrated-mesh (c-mesh)"; "Data transfers over the c-mesh are also statically scheduled and guaranteed to not conflict with other data packets." |

### 1.6 POWER

| quantity | value | unit | evidence_kind | citation | quote / arithmetic |
|---|---|---|---|---|---|
| **Total chip power** | **65.8** | W | published | Table I "Chip Total" | "Chip Total … 65.8 W". Restated §VIII-B: "An ISAAC chip consumes more power (65.8 W) than a DaDianNao chip (20.1 W) of the same size" |
| 168-tile array power | 55.4 | W | published | Table I | "168 Tile Total … 55.4 W" |
| HyperTransport power | 10.4 | W | published | Table I | "Hyper Tr … 10.4 W" — and §VIII-B: "The HT is a constant overhead of 10 W … but only 16 % of ISAAC chip power" |
| Per-tile power | 330 | mW | published | Table I | "1 Tile Total … 330 mW" |
| **ADC power, per instance** | **2.0** | mW | derived | Table I ADC row | 16 mW ÷ 8 ADCs = **2 mW** per 8-bit 1.2 GSps ADC |
| **ADC power, whole chip** | **32.26** | W (**49.0 % of 65.8 W**) | derived | Table I | 168 tiles × 12 IMAs × 16 mW = **32.256 W**. Matches the paper's framing: "the ADCs accounting for nearly half the chip power" (§X) ✓ |
| DAC power, per instance | 3.91 | µW | derived | Table I DAC row | 4 mW ÷ 1024 = **3.906 µW** |
| Memristor-array power, per array | 0.300 | mW | derived | Table I array row | 2.4 mW ÷ 8 = **0.3 mW** per 128×128 array |
| S+H power, per instance | 9.77 | nW | derived | Table I S+H row | 10 µW ÷ 1024 = **9.766 nW** (cited cell is 10 nW ✓) |

**Component power breakdown, per tile (330 mW total)** — fully transcribed, group totals:

| component | per-tile power | % of tile | source |
|---|---|---|---|
| **ADC** (96 instances) | **192 mW** | **58.2 %** | 12 × 16 mW; paper states "58 %" |
| eDRAM buffer | 20.7 mW | 6.3 % | Table I |
| Memristor arrays (96) | 28.8 mW | 8.7 % | 12 × 2.4 mW |
| DAC (12 288) | 48 mW | 14.5 % | 12 × 4 mW |
| IR (12 × 2 KB) | 14.88 mW | 4.5 % | 12 × 1.24 mW |
| Router (¼ share) | 10.5 mW | 3.2 % | 42/4 mW |
| eDRAM-to-IMA bus | 7 mW | 2.1 % | Table I |
| IMA OR (12 × 256 B) | 2.76 mW | 0.8 % | 12 × 0.23 mW |
| IMA S+A (48) | 2.4 mW | 0.7 % | 12 × 0.2 mW |
| Tile OR (3 KB) | 1.68 mW | 0.5 % | Table I |
| Sigmoid (2) | 0.52 mW | 0.16 % | Table I |
| MaxPool (1) | 0.4 mW | 0.12 % | Table I |
| S+H (12 288) | 0.12 mW | 0.04 % | 12 × 10 µW |
| Tile S+A (1) | 0.05 mW | 0.02 % | Table I |
| **Σ** | **329.81 mW** | 99.9 % | vs printed 330 mW ✓ |

> Supporting quote (`shafiee2016isaac`, §VIII-A, p.10): "Table I shows that the ADCs account
> for 58 % of tile power and 31 % of tile area. No other component takes up more than 15 %
> of tile power. … The eDRAM buffer and the eDRAM-IMA bus together take up 47 % of tile
> area. Many of the supporting digital units (shift-and-add, MaxPool, Sigmoid, SRAM buffers)
> take up negligibly small amounts of area and power."

**Component area breakdown, per tile (0.372 mm² total):**

| component | per-tile area (mm²) | % of tile |
|---|---|---|
| **ADC** (96) | **0.1152** | **31.0 %** |
| eDRAM-to-IMA bus | 0.090 | 24.2 % |
| eDRAM buffer | 0.083 | 22.3 % |
| Router (¼ share) | 0.03775 | 10.1 % |
| IR (12 × 2 KB) | 0.0252 | 6.8 % |
| Tile OR (3 KB) | 0.0032 | 0.86 % |
| IMA OR (12) | 0.00924 | 2.48 % |
| IMA S+A (48) | 0.00288 | 0.77 % |
| Memristor arrays (96) | 0.0024 | 0.65 % |
| DAC (12 288) | 0.00204 | 0.55 % |
| Sigmoid (2) | 0.0006 | 0.16 % |
| S+H (12 288) | 0.00048 | 0.13 % |
| MaxPool | 0.00024 | 0.06 % |
| Tile S+A | 0.00006 | 0.02 % |
| **Σ** | **0.37229** | 100.1 % ✓ |

### 1.7 Headline efficiency metrics (Table IV, verbatim)

| Architecture | CE (GOPs/(s·mm²)) | PE (GOPs/W) | SE (MB/mm²) |
|---|---|---|---|
| DaDianNao | 63.46 | 286.4 | 0.41 |
| **ISAAC-CE** | **478.95** | **627.5** | **0.74** |
| ISAAC-PE | 409.67 | 644.2 | 0.62 |
| ISAAC-SE | 103.35 | 312.5 | 20.25 |

Caption: "TABLE IV — COMPARISON OF ISAAC AND DADIANNAO IN TERMS OF CE, PE, AND SE.
HYPERTRANSPORT OVERHEAD IS INCLUDED."

Metric definitions (§VII "Metrics"): "CE: Computational Efficiency is represented by the
number of 16-bit operations performed per second per mm² (GOPS/s × mm²)"; "PE: Power
Efficiency is represented by the number of 16-bit operations performed per watt (GOPS/W)";
"SE: Storage Efficiency is the on-chip capacity for synaptic weights per unit area (MB/mm²)".

**Derived chip throughput (needed to validate the conversion model):**
478.95 GOPS/(s·mm²) × 85.4 mm² = **40 902 GOPS = 40.9 TOPS = 20.45 T-MAC/s** (2 ops per MAC).

---

## 2. UNSOURCED list

Reported as **UNSOURCED**, not estimated, unless an explicit estimate is labelled.

| vocabulary key | why unsourced |
|---|---|
| `t_hop` | Router **latency** is never stated. Table I gives flit size (32 b), port count (8) and power (42 mW) only. A 1-cycle-per-hop assumption at 1.2 GHz would give 0.833 ns, but the paper does not support it. |
| `e_inter_tile_hop` | Only router *power* is published; there is **no stated activity factor or utilisation**. The 4.375–5.25 pJ band in §1.3 is explicitly `estimated`. |
| `e_neuron_update` | ISAAC has no per-neuron soma. The sigmoid unit's **throughput per cycle is never stated**, so 52 pJ/tile-cycle cannot be divided into a per-neuron figure. |
| `area_per_neuron_logic` (as a true per-neuron cost) | Same reason. The 300 µm² sigmoid-unit area in §1.2 is a *shared unit*, and its multiplicand is tiles, not `neurons_physical`. |
| `area_per_state_bit`, `membrane_bits`, `e_leak_per_neuron_step` | ISAAC is a **non-spiking** CNN/DNN accelerator with no membrane state. Not applicable rather than missing. |
| **Supply voltage** | Never stated anywhere in the paper. Neither a core Vdd nor a crossbar read voltage is given (only "for the input voltages we are considering, i.e., DAC output voltage range"). |
| **Operating temperature** | Never stated. Thermal noise is mentioned only qualitatively, via ref [26]. |
| **`conductance_levels` (absolute)** | Only *bits per cell* = 2 (⇒ 4 levels) is given. No R_on/R_off, no absolute conductance values. |
| **`write_sigma` / `read_sigma`** | No numeric device-variability figures. The paper defers entirely to Hu et al. [26]: "Hu et al. [26] demonstrate 5 bits per cell and a 512×512 crossbar array showing no accuracy degradation compared to a software approach for the MNIST dataset, after considering thermal noise in memristor, short noise in circuits, and random telegraphic noise in the crossbar." |
| **Weight-programming energy/time** (`e_core_program`, `t_program_per_byte`, `e_dma_per_byte` for weight load) | The paper explicitly excludes it: weights are loaded once, and no programming cost is modelled ("After training has determined the weights for every neuron, the weights are appropriately loaded into memristor cells with a programming step"). No number given. |
| `e_sync_barrier`, `t_sync_barrier` | No barrier exists — the pipeline is statically scheduled by FSMs. Not applicable. |
| `p_static_per_core`, `p_static_global` | No static/leakage decomposition. Table I powers are undifferentiated. The 10.4 W HT is the closest thing to a static term (paper calls it "a constant overhead"). |
| **Per-cell area at >1 significant figure** | Table I prints `0.0002` mm² — one sig fig. The derived 1.526 × 10⁻³ µm²/cell inherits that precision (true value anywhere in ≈[1.15, 1.91] × 10⁻³ µm² for a 0.00015–0.00025 rounding window). |
| **ISAAC-PE / ISAAC-SE geometry** | Only ISAAC-CE's parameters are tabulated. Figs. 5a/5b identify PE and SE design points but their (arrays, ADCs, IMAs) tuples are not stated in text. |

---

## 3. Validity domain

### 3.1 What these numbers are, and what produced them

**ISAAC was never fabricated.** Every constant is a model output. The model stack, verbatim
from §VII "Energy and Area Models" (p.7):

> "All ISAAC parameters and their power/area values are summarized in Table I. We use CACTI
> 6.5 [45] at 32 nm to model energy and area for all buffers and on-chip interconnects. The
> memristor crossbar array energy and area model is based on [26]. The energy and area for
> the shift-and-add circuits, the max-pool circuit, and the sigmoid operation are adapted
> from the analysis in DaDianNao [9]. For off-chip links, we employ the same HyperTransport
> serial link model as that used by DaDianNao [9]."

> "For ADC energy and area, we use data from a recent survey [46] of ADC circuits published
> at major circuit conferences. For most of our analysis, we use an 8-bit ADC at 32 nm that
> is optimized for area. … An SAR ADC has four major components [36]: a vref buffer, memory,
> clock, and a capacitive DAC. To arrive at power and area for the same style ADC, but with
> different bit resolutions, we scaled the power/area of the vref buffer, memory, and clock
> linearly, and the power/area of the capacitive DAC exponentially [59]."

> "For most of the paper, we assume a simple 1-bit DAC because we need a DAC for every row in
> every memristor array. To explore the design space with multi-bit DACs, we use the
> power/area model in [59]."

Underlying circuit sources: ADC survey = **Murmann, "ADC Performance Survey 1997–2015"** [46];
the SAR ADC style = **Kull et al., "A 3.1 mW 8b 1.2 GS/s Single-Channel Asynchronous SAR ADC
… in 32 nm Digital SOI CMOS", JSSC 2013** [36]; DAC model = **Saberi et al.** [59]; S&H =
**O'Halloran & Sarpeshkar, "A 10-nW 12-bit Accurate Analog Storage Cell"** [48]; crossbar =
**Hu et al., "Dot-Product Engine…", DAC-53 2016** [26].

**Performance model** (§VII "Performance Model", p.8) — *analytical, not cycle-accurate*:

> "We have manually mapped each of our benchmark applications to the IMAs, tiles, and nodes
> in ISAAC. … This gives us a deterministic execution model for ISAAC and the
> latency/throughput for a given CNN/DNN can be expressed with analytical equations. …
> CNNs/DNNs executing on these tiled accelerators do not exhibit any run-time dependences or
> control-flow, i.e., cycle-accurate simulations do not capture any phenomena not already
> captured by our analytical estimates."

### 3.2 Validity envelope

| axis | value |
|---|---|
| Technology node | 32 nm (CACTI 6.5 + 32 nm ADC survey data) |
| Supply voltage | **not stated** |
| Temperature | **not stated** |
| Clock (digital) | 1.2 GHz |
| Compute cycle | 100 ns (crossbar read); sensitivity point at 200 ns |
| Workload class | **Inference only.** "The architecture is not used for in-the-field training; it is only used for inference" (§III) |
| Numeric format | 16-bit fixed point. 32-bit sensitivity: "This would reduce overall throughput by 4×" (§VIII-A) |
| Algorithm constraint | **No LRN/LCN layers.** "LRN layers are not amenable to acceleration with crossbars" (§II-B) |
| Utilisation caveat | Peak CE/PE assume full utilisation: "This sub-section reports peak CE, PE, and SE values, assuming that all IMAs can be somehow utilized in every cycle." Real benchmarks fall well short — "in some benchmarks, the first layer has to be replicated more than 50K times to keep the last layer busy in every cycle. Since we don't have enough storage for such high degrees of replication, the last classifier layers also see relatively low utilization in the IMAs" (§VIII-B) |
| Analog-noise treatment | **Not modelled quantitatively.** Deferred to [26]; "A marginal increase in signal noise can be endured given the inherent nature of CNNs to tolerate noisy input data" (§VIII-A) |

### 3.3 Internal disagreements — FLAGGED

**(a) The 1707 GOPS/(s·mm²) bare-array claim contradicts Table I's array area.**

§VIII-B, p.10: *"a 128×128 memristor array with 2 bits per cell has a CE of 1707 GOPS/s × mm²"*.

A 128×128 array with w=2 delivers 128 rows × 16 logical 16-bit weights = 2048 MACs = 4096 ops
per 16-cycle (1.6 µs) burst → **2.56 GOPS**. For CE = 1707, the implied array area is
2.56 / 1707 = **1.50 × 10⁻³ mm² = 1500 µm²** — **60 × larger** than the 25 µm² that Table I's
`0.0002 mm² ÷ 8` yields.

Table I is the trustworthy side: it reproduces *every* other published aggregate exactly
(§1.0's five reconciliations, plus chip totals and peak CE/PE below). The 1707 figure
reconciles with nothing. Given the paper's own October-2016 erratum banner, treat it as a
**stale/erroneous figure** and do not use it. Recorded here only so a later reader does not
"correct" the profile toward it.

**(b) ADC sample rate: 1.2 GSps (Table I) vs 1.28 GSps (§V text).**
The text's 1.28 GSps is the *architecturally required* rate (128 columns ÷ 100 ns); Table I's
1.2 GSps is the *nameplate* of the surveyed Kull et al. ADC. A 6.7 % shortfall. Both are
carried as the `e_adc_conversion` / `t_adc_conversion` band. Prefer 1.28 GSps for dataflow
counting and 1.2 GSps for circuit-level costing.

**(c) `area_per_cell` implies a ~39 nm crossbar pitch at a 32 nm node, with a 1T1R cell.**
The derived 1526 nm²/cell = 4F² at F = 19.5 nm. But §II-D states: "The crossbar is
implemented with a 1T1R cell structure to facilitate more precise writes to memristor cells".
A 1T1R cell at 32 nm cannot be 4F² at F = 19.5 nm — an access transistor alone would exceed
it. Either the crossbar model of [26] assumes a sub-lithographic memristor pitch, or the
`0.0002` entry is rounded/optimistic. Flagged; the number is transcribed as-is because it is
the one Table I's totals are built from.

### 3.4 What DOES reconcile (confidence anchors)

| check | computed | paper |
|---|---|---|
| Peak chip throughput | 16 128 arrays × 4096 ops ÷ 1.6 µs = **41 288 GOPS** | 478.95 × 85.4 = **40 902 GOPS** (0.9 %) |
| Peak PE | 41 288 GOPS ÷ 65.8 W = **627.5 GOPS/W** | Table IV: **627.5** ✓ exact |
| Chip ADC power via the conversion model | 20.45 T-MAC/s × 1 conv/MAC × 1.5625 pJ = **31.95 W** | 168 × 12 × 16 mW = **32.26 W** (1.0 %) |
| Chip area | 168 × 0.372 + 22.88 = **85.38 mm²** | **85.4 mm²** ✓ |
| Chip power | 168 × 0.330 + 10.4 = **65.84 W** | **65.8 W** ✓ |

The peak-PE match to four significant figures independently confirms both the op-counting
convention (2 ops per 16-bit MAC) and the conversion model of §4.

---

## 4. CONVERSION MODEL

### 4.1 The dataflow, from the paper

1. **Weights are bit-sliced across columns.** §V: *"We therefore represent one 16-bit
   synaptic weight with 16/w w-bit cells located in the same row."* → `cells_per_weight =
   weight_bits / cell_bits = 16/2 = 8` **adjacent columns per logical weight**.
2. **Inputs are bit-sliced across time.** §V: *"we provide 16 voltage levels sequentially,
   where voltage level i is a 0/1 binary input representing bit i of the 16-bit input
   number. … all 16 bits of the input have been handled in 16 cycles."* → `input_bits /
   dac_bits = 16/1 = 16` **cycles per dot product**.
3. **Every column is converted every cycle.** §V: *"a 128×128 crossbar may produce 128
   bitline currents every 100 ns … these bitline currents are latched in 128 sample-and-hold
   circuits. In the next 100 ns cycle, these analog values in the sample-and-holds are fed
   sequentially to a single 1.28 GSps ADC unit."* → **128 conversions per array per cycle**,
   with `adc_sharing_factor = 128` and 1 ADC per array.

### 4.2 The formula

For a logical layer computing `N` outputs from a length-`L` dot product (`macs = L × N`):

```
arrays            = ceil(L / array_rows) * ceil(N * cells_per_weight / array_cols)
cycles_per_read   = input_bits / dac_bits
conversions/array = array_cols        (every column, every cycle)

adc_conversions   = arrays * cycles_per_read * array_cols
```

Dropping the ceilings (large-layer limit) this collapses to the closed form:

```
adc_conversions = macs * (weight_bits / cell_bits) * (input_bits / dac_bits) / array_rows
                = macs * cells_per_weight * (input_bits / dac_bits) / array_rows
```

**For ISAAC-CE** (`weight_bits=16, cell_bits=2, input_bits=16, dac_bits=1, array_rows=128`):

```
adc_conversions = macs * 8 * 16 / 128 = macs * 1.0
```

> **Exactly one 8-bit ADC conversion per 16-bit MAC.** Add ×129/128 (+0.78 %) if the unit
> column is charged (§1.5).

Converter population:

```
adc_count = arrays * ceil(array_cols / adc_sharing_factor)
          = arrays * ceil(128 / 128) = arrays          (1 ADC per array; 16 128 chip-wide ✓)
```

### 4.3 Reasoning, and why the two bit-slicing factors both appear

The naive intuition — "`array_rows` MACs make one column integration, and a bit-serial input
converts it once per input slice" — captures factor (2) but **misses factor (1)**. In ISAAC a
single logical 16-bit weight occupies **8 physical columns**, each of which is separately
integrated *and separately converted*; the 8 partial results are merged afterwards by the
shift-and-add tree, in the digital domain. Omitting `cells_per_weight` therefore undercounts
conversions by exactly 8×.

**Numeric proof** (§3.4, row 3): the formula predicts 20.45 T conversions/s, which at
1.5625 pJ gives **31.95 W** of chip ADC power against Table I's **32.26 W** — a 1.0 % match.
The 8×-lower alternative would predict 4.0 W, i.e. 6 % of chip power, flatly contradicting the
paper's own "the ADCs account for 58 % of tile power".

### 4.4 ⚠ Disagreement with the in-repo model — see §5.3

The repo's `bit_sliced_crossbar` model computes
`adc_conversions = ceil(macs / array_rows) * input_bits`, which lacks the `cells_per_weight`
factor and therefore under-counts ISAAC by **8×**.

---

## 5. In-repo cross-check

### 5.1 Files carrying ISAAC constants

| file | what it holds |
|---|---|
| `src/mimarsinan/mapping/platform/imc_platforms_literature.py` (lines 60–71) | `isaac_like` IMCPlatform geometry + provenance quote |
| `src/mimarsinan/deployment_record/platform_physics/conversion.py` (lines 43–68, 96–103) | `bit_sliced_crossbar` conversion model, whose docstring names ISAAC |
| `/home/yigit/backups/se_review_copy/papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json` | the extraction card the registry entry is transcribed from (outside this repo) |
| `src/mimarsinan/deployment_record/platform_physics/profiles/` | **no `isaac.json` exists yet** (only `truenorth`, `loihi`, `generic_estimated_22nm`) |

### 5.2 Value-by-value cross-check of `isaac_like`

| repo field | repo value | paper value | verdict |
|---|---|---|---|
| `max_axons` (crossbar rows) | 128 | 128 | ✅ AGREES |
| `max_neurons` (crossbar cols) | 128 | 128 physical columns | ⚠ **see finding F1** |
| `count` | 16 128 | 8 × 12 × 168 = 16 128 | ✅ AGREES (arithmetic verified) |
| `weight_bits` | 16 | 16-bit fixed point = 8 cells × 2 bits | ✅ AGREES with the card's own contract (`bits_per_cell × cells_per_weight`) |
| `has_bias` | False | ISAAC has a *unit column* for bias/encoding correction, and §III-E-equivalent text writes bias into the array | ⚠ **see finding F2** |
| provenance quote | *"The optimal design point has 8 128×128 arrays, 8 ADCs per IMA, and 12 IMAs per tile. We refer to this design as ISAAC-CE."* | **verbatim match** in §VIII-A "Design Space Exploration" | ✅ quote is exact |
| provenance location | "Section VII (Results, 'Design Space Exploration'), p.9 col2" | The sentence is in **§VIII (Results)**, not §VII (§VII is Methodology). Page 9 col 2 is correct. | ⚠ **see finding F3** (minor) |
| `claim_eligibility` | `curve-only` | — | ✅ appropriate: a simulated architecture, not silicon |

### 5.3 Findings

**F1 — `max_neurons = 128` double-counts against `weight_bits = 16` (geometry inconsistency).**
The card sets `weight_bits = 16`, whose contract is "`bits_per_cell × cells_per_weight`" =
2 × 8. But it also sets `max_neurons = 128`, i.e. one neuron per *physical column*. Both
cannot hold: if a weight consumes 8 columns, a 128-column array holds **16** logical 16-bit
neurons, not 128. As written the entry claims 8× more 16-bit neurons per array than ISAAC has.
The existing note acknowledges the abstraction — *"our model treats a neuron as one column
(bit-slicing abstracted into weight_bits)"* — but the two fields are then mutually
inconsistent rather than merely approximate. **Self-consistent options:** `(128 rows × 16 cols,
weight_bits=16)`, or `(128 × 128, weight_bits=2)` with cells rather than weights as the unit.

**F2 — `has_bias=False` is arguable.** ISAAC dedicates an extra "unit column" per array
(§V: *"the columns per array has grown from 128 to 129"*) and folds the weight bias (2¹⁵) and
flipped-encoding correction through it. That is not a per-neuron additive bias in the SNN
sense, so `False` is defensible, but the 129th column is a real, published structure the
geometry currently drops. Low severity.

**F3 — provenance section number off by one.** The quote is in **§VIII-A**, not §VII.
Cosmetic, but the card's stated rule is exact transcription.

**F4 — ⚠ HIGH: `bit_sliced_crossbar` under-counts ISAAC conversions by 8×.**
`conversion.py:59-62` computes `ceil(macs / array_rows) * input_bits`. Its docstring claims
this is *"The dataflow ISAAC and PRIME describe"* (line 99–100). It is not: the model omits
`cells_per_weight = weight_bits / cell_bits`, the *weight*-side bit-slicing that gives the
model its name. For ISAAC-CE the true count is `macs`; the model returns `macs/8`. Pricing at
1.5625 pJ this reports **4.0 W** of chip ADC power instead of **32.26 W** — turning the
paper's dominant cost (58 % of tile power) into a rounding error.

*Two remedies (no code change is in scope for this task, so both are recorded, not applied):*
- **Preferred:** add a `cells_per_weight` (or `weight_bits`+`cell_bits`) parameter to the
  model and multiply it in. This also fixes PRIME (which is under-counted 2×).
- **Declaration-level workaround with the model exactly as it stands:** declare
  `input_bits = 128` (= 16 input bits × 8 cells/weight) with `array_rows = 128`, which yields
  `ceil(macs/128) × 128 = macs` ✓. Correct arithmetic, but the parameter no longer means what
  it is named — it must be commented if used.

**F5 — `adc_count` in the repo model is CORRECT for ISAAC.**
`arrays * ceil(array_cols / adc_sharing_factor)` = `arrays * ceil(128/128)` = 1 ADC per array,
matching Table I's "8 ADCs / 8 arrays per IMA" and 16 128 ADCs chip-wide. No change needed.

**F6 — no ISAAC physics profile exists yet.** `platform_physics/profiles/` has no
`isaac.json`. The constants in §1 are what such a profile would carry; the honest
`evidence_kind` for nearly all of them is `derived` (from Table I group totals) or
`published` (Table I directly), never `datasheet`.

---

## 6. Citations (not yet injected)

Injection into a real `.bib` is **deferred**. The only `.bib` in this repo is
`sana_fe/references.bib`, which belongs to the vendored SANA-FE tree and must stay untouched;
no other `.bib` exists here and none was created.

The BibTeX below was **generated by the `asta-papers` tooling** (`semantic_scholar_search` →
`inject_papers_to_bib` into a scratchpad file, then copied verbatim). It was not hand-written.
When a real bibliography exists, re-run `inject_papers_to_bib` with id `shafiee2016isaac`
rather than pasting this block.

```bibtex
@Article{shafiee2016isaac,
 author = {A. Shafiee and Anirban Nag and N. Muralimanohar and R. Balasubramonian and J. Strachan and Miao Hu and R. S. Williams and Vivek Srikumar},
 booktitle = {International Symposium on Computer Architecture},
 journal = {2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA)},
 pages = {14-26},
 title = {ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars},
 year = {2016},
 doi = {10.1145/3007787.3001139},
 url = {https://doi.org/10.1145/3007787.3001139},
}
```

Cached id for re-injection: `shafiee2016isaac`.

**Secondary circuit sources** cited *by* ISAAC for its ADC/DAC/S&H/crossbar models (not yet
retrieved or cached; listed so the model provenance chain is traceable):
ref [26] Hu et al., *Dot-Product Engine for Neuromorphic Computing*, DAC-53 2016;
ref [36] Kull et al., *A 3.1 mW 8b 1.2 GS/s Single-Channel Asynchronous SAR ADC …in 32 nm
Digital SOI CMOS*, JSSC 2013;
ref [45] Muralimanohar et al., *CACTI 6.0*, MICRO 2007;
ref [46] Murmann, *ADC Performance Survey 1997–2015*;
ref [48] O'Halloran & Sarpeshkar, *A 10-nW 12-bit Accurate Analog Storage Cell*, JSSC 2004;
ref [59] Saberi et al., *Analysis of Power Consumption and Linearity in Capacitive DACs used
in SAR ADCs*, 2011;
ref [9] Chen et al., *DaDianNao*, MICRO-47 2014.
