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
- **`t_cycle` is one synaptic operation**, because on ODIN it is: "each SOP takes two
  clock cycles to complete", so 2 / 75 MHz = 26.667 ns. There is no separate network
  timestep at this level of the architecture.
- **`area_per_cell` (0.68 µm²/synapse) is declared but never summed** with the
  aggregate `area_per_core_total`. It is the only per-cell area among the three
  correlated targets that can be checked against a published total: 65,536 × 0.68 µm²
  = 51.6 % of the core, leaving roughly half for the neurons, the SDSP update logic,
  the scheduler and the controller.

## The programming group — DERIVED, and how (TS6)

Frenkel reports no weight-load energy and no per-byte configuration rate, so all three
programming constants are **authored derivations** (`evidence_kind: derived`), each
naming its anchor and arithmetic, each with its nominal at the band's log-space centre
(`sqrt(low x high)`):

- **`e_program_per_byte` = 8.43 / 20.649 / 50.58 pJ/B.** An ODIN synapse is 4 bits
  (3-bit weight + 1 mapping bit), so a byte of synaptic SRAM holds two synapses and the
  per-byte access anchor is `2 x e_mac` = 16.86 pJ/B. An SRAM write costs ~1–3× its
  read → band `[0.5x, 3x]`. `e_mac` already contains the weight read *and* update, the
  neuron state access and the controller overheads, so it overstates the bare SRAM
  access: that is what the 0.5× corner is for.
- **`t_program_per_byte` = 5.09 / 285.4 / 1272.7 ns/B.** Donor-scaled along
  `validity.technology_node_nm` from TrueNorth's scan chain (800 ns/B, same 28 nm node)
  and the generic 22 nm exemplar (4/80/1000 → 5.09/101.8/1272.7). Band = the envelope
  of both scaled donors; nominal = the geometric mean of their scaled nominals. Two
  orders of magnitude wide because the donors genuinely disagree by that much.
- **`e_core_program` = 1.079 / 2.158 / 4.316 nJ.** No `e_core_init` exists here, so the
  anchor is the **core-state sweep** the geometry defines: 256 neurons × `e_mac` =
  2.158 nJ. Band `[0.5x, 2x]`. ODIN is a single core, so a program load pays this once.

## What this profile deliberately does not declare

The whole `interconnect` group: ODIN is a **single core** with no NoC, so hop and
packet energies are not absent-for-lack-of-data, they are structurally zero — and a
multi-core deployment mapped onto "ODIN" would be a different chip, not this one.

Also absent: the `host` group (a property of the deployment host), `e_dma_per_byte`
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
