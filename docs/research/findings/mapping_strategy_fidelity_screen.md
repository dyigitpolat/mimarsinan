# Mapping-strategy fidelity screen — the `mapping_strategy` axis collapse (Wave-2 A3)

**Axis:** `mapping_strategy` (`{packed, identity, neuron_split, coalesced}`)
**Flip:** `ASSERTED_UNSCREENED` → `SCREENED_COLLAPSED` (representative `packed`)
**Artifact (the lock that proves it):**
`tests/integration/test_torch_sim_fidelity.py` (+ its harness `_torch_sim_fidelity.py`).
**Captured pass:** `34 passed` (run command below).

## Scope (the load-bearing distinction): FIDELITY-ONLY

`mapping_strategy` is a **FAITHFULNESS axis** — the strategies are different *packings*
of the **same deployment contract** onto hard cores. Equivalent packings must compute the
**same deployed value**: a disagreement is a packing **BUG**, not an axis *interaction*.
So the collapse rests on a measured **bit-exact FIDELITY** artifact (like the
`encoding_placement` and `backend` precedents) — **NOT** a semantic GPU equivalence
screen.

The collapse is **FIDELITY-ONLY (deployed value)**. It does **NOT** collapse:

- **CAPABILITY** — which packing a chip can run (coalescing is a chip capability:
  inter-core membrane transfer; with `allow_coalescing=False` a wide fan-in is
  *unmappable*, raising `WideFanInUnsupportedError`, not silently lossy).
- **COST / UTILIZATION** — cores consumed, axon budget, crossbar count per strategy.
  Recorded as a frontier.

## What the lock proves (BIT-EXACT deployed value)

`test_torch_sim_fidelity.py` builds, from one converted model, the torch
neuromorphic-forward (NF) and the deployed HCM sim for every
**(mode × mapping-config × model)** cell and asserts the strongest valid invariant:

- **BIT-EXACT** (`float64 atol=0`; LIF additionally **per-neuron `k == k`** spike counts)
  for `identity` / `neuron_split` / `axon_fuse` across every bit-exact mode
  (`lif`, `ttfs_cycle_based`, `ttfs_quantized`) and every model topology (single-core,
  multi-core, **sync-point** with a mid-graph LayerNorm ComputeOp).
  - `identity` — one un-split core per perceptron.
  - `neuron_split` — a core's neurons tiled across hard cores, reassembled bit-exact.
  - `axon_fuse` — **the partial-sum mapping (coalescing)**: a fan-in wider than one hard
    core consumes N hard cores fused into one *wider* crossbar; the full weighted sum is
    computed once, so it is bit-exact (membrane potentials in one merged core, never
    re-fired spikes).
- The harness `assert_config_triggered` fails loudly if a config did not actually pack the
  way its name claims — so no cell can vacuously pass as "identity in disguise".

The bounded `ttfs` (continuous) cell is excluded from the bit-exact set by construction
(real-valued NF vs S-step quantized sim) and is not part of this collapse.

## The `coalescing` caveat — VALUE_DOMAIN_ONLY attribution (GAP-1)

`coalescing` is **value-domain bit-exact** and **spike-conserving**, but its per-neuron
**ATTRIBUTION** reassembly is historically cracked at VGG scale (**GAP-1**:
`coalescing + neuron_split at VGG scale` scrambles ~2% of the per-neuron attribution
while the deployed accuracy stays exact). The coverage instrument records this as
`AttributionFidelity.VALUE_DOMAIN_ONLY` (`KNOWN_CRACKED_REGIONS` in `coverage_ledger.py`).
The collapse therefore claims **only the deployed VALUE** for `coalesced`; it does **not**
over-claim per-neuron attribution.

## Reproduce

```
PYTHONPATH=<repo>/src:<spikingjelly>:tests \
  <env>/bin/python -m pytest tests/integration/test_torch_sim_fidelity.py -q
# => 34 passed
```

## Conclusion

Equivalent packings of one contract agree on the deployed value **bit-exactly** across
every bit-exact mode × model. The `mapping_strategy` axis therefore collapses to a single
representative (`packed`) for **fidelity** — with `coalesced` per-neuron attribution
explicitly bounded to `VALUE_DOMAIN_ONLY`, and cost/utilization left a **frontier**.
