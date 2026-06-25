# Backend cross-simulator parity screen — the `backend` axis collapse (Wave-2 A2)

**Axis:** `backend` (`{nevresim, sanafe, hcm, lava}`)
**Flip:** `ASSERTED_UNSCREENED` → `SCREENED_COLLAPSED` (representative `sanafe`)
**Instrument:** `mimarsinan.chip_simulation.cross_sim_parity` (Wave-1 B1), run LIVE.
**Machine-readable artifact:** [`backend_cross_sim_screen.json`](backend_cross_sim_screen.json)
(passes `assert_cross_sim_screen_sound(..., justifies_collapse=True)`).

## Scope (the load-bearing distinction): FIDELITY-ONLY

`backend` is a **FAITHFULNESS axis** — the candidate backends are different *simulators*
of the **same deployment contract**. Faithful simulators of one contract must agree on
the **deployed value**: a disagreement is a **BUG**, not an axis *interaction*. So the
collapse rests on a measured **PARITY/FIDELITY** artifact, exactly like the
`encoding_placement` (offload≡subsume) precedent — **NOT** a semantic GPU equivalence
screen.

The collapse is **FIDELITY-ONLY (deployed value)**. It does **NOT** collapse:

- **CAPABILITY** — which backend×mode actually *runs* (e.g. lava is LIF-only; SANA-FE /
  nevresim need their binaries). Recorded as a frontier (see the INAPPLICABLE rows).
- **COST / UTILIZATION** — per-backend energy, cores, NoC traffic. Recorded as a
  frontier (the `cost_extraction.CostRecord.backend` cost coordinate keeps backends
  distinct cells for cost).

This is the same fidelity-only caveat the `encoding_placement` collapse carries
(`PROGRAM_PLAN_v2.md §E5 / cost caveat`).

## What was measured (LIVE)

`build_torch_and_hcm` converts a representative model (encoding-host fc + two on-chip
layers) and runs, for each cell, the torch neuromorphic-forward (NF) reference — the
domain of the nevresim Python sim path — against the deployed **HCM/SCM** analytical sim.
The per-perceptron deployed-**VALUE** records are compared by
`cross_sim_parity.measured_max_abs_diff`, which **wraps**
`pipelining.core.nf_scm_parity.compare_normalized_records` (order-insensitive
per-perceptron sorted-multiset comparison, `atol=0`). The parity math is reused, never
re-implemented.

Cells span **both modes** (`lif`, `ttfs_cycle_based`) × **both packings** (`identity`,
`neuron_split`):

| cell | backend pair | state | measured `max_abs_diff` |
|---|---|---|---|
| `lif/identity@T8` | nevresim ↔ hcm | **AGREE** | **0.0** |
| `lif/neuron_split@T8` | nevresim ↔ hcm | **AGREE** | **0.0** |
| `ttfs_cycle_based/cascaded/identity@T8` | nevresim ↔ hcm | **AGREE** | **0.0** |
| `ttfs_cycle_based/cascaded/neuron_split@T8` | nevresim ↔ hcm | **AGREE** | **0.0** |
| `ttfs_cycle_based/cascaded/identity@T8` | hcm ↔ **lava** | **INAPPLICABLE** | — (LIF-only capability gap) |
| `ttfs_cycle_based/cascaded/neuron_split@T8` | hcm ↔ **lava** | **INAPPLICABLE** | — (LIF-only capability gap) |

Every applicable pair AGREEs at **`max_abs_diff = 0.0`** (bit-exact deployed value).
`lava` is screened by CAPABILITY only — DERIVED from the `_BACKEND_CAPS` registry via
`SpikingModePolicy.supports_backend`; it is **never executed-to-hang**.

## Corroborating multi-backend parity LOCKS (the validated corner)

The LIVE screen above is reinforced by the program's standing multi-backend parity
locks, which establish **nevresim / sanafe / lava ≡ HCM** in the validated corner. Those
that touch a compiled binary are env-gated (skip when the binary is absent), so they do
not block CI here, but they are the cited evidence:

- `tests/integration/test_scm_hcm_sim_parity.py` — SCM and HCM produce **bit-identical**
  `SpikingHybridCoreFlow` output in every spiking mode (lif/ttfs/ttfs_quantized). *Not
  binary-gated.*
- `tests/integration/test_nf_hcm_per_node_spike_parity_mmixcore.py` — exact **per-neuron
  LIF** spike-count parity (torch NF node == HCM hard-core neuron). *Not binary-gated.*
- `tests/unit/pipelining/pipeline_steps/test_nf_scm_parity_gate.py` — the **NF↔SCM
  per-neuron** parity gate for analytic TTFS schedules. *Not binary-gated.*
- `tests/unit/chip_simulation/nevresim/test_execute_simulator.py` — **nevresim
  execute-parity** vs HCM connectivity. *Gated on the nevresim tree + a C++20 compiler.*
- `tests/integration/test_sanafe_hcm_parity.py` — **SANA-FE per-core** input/output
  spike parity vs HCM. *Gated on `import sanafe` + the LIF MNIST artifact.*
- `tests/integration/test_loihi_hcm_spike_parity.py` — **Lava (Loihi) LIF** per-core
  spike parity vs HCM. *Gated on `import lava` (+ `@pytest.mark.slow`).*

## Conclusion

Faithful backends agree on the deployed value to `max_abs_diff = 0.0` on the screened
cells, and the standing locks extend that to nevresim/sanafe/lava in the validated
corner. The `backend` axis therefore collapses to a single representative (`sanafe`) for
**fidelity** — capability (lava LIF-only; binary availability) and cost/utilization stay
**frontiers**, not collapsed.
