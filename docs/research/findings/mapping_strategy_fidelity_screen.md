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

## The `coalescing` caveat — VALUE_DOMAIN_ONLY attribution (GAP-1, sharpened by Wave-2 C3)

`coalescing` is **value-domain bit-exact** and **spike-conserving**, but its per-neuron
**ATTRIBUTION** reassembly is cracked at VGG scale (**GAP-1**: under
coalescing + output-tiling the IR-id order decouples from the output-slice order once
IR-graph compaction reorders ids, so an `sorted(ir_id)` concatenation scrambles ~2% of
the per-neuron attribution while the deployed accuracy stays exact).

**Wave-2 C3 reconciliation (this is a SHARPEN, not a RESOLVE):** C3 fixed the
**fidelity-HARNESS** reassembler — the joint `(perceptron_output_slice, ir_id)` keying in
`tests/integration/_split_reassembly.py` (and the same keying in
`nf_scm_parity._group_record_by_perceptron`) makes coalescing+output-tiling per-neuron
attribution **bit-exact in the harness**, locked by
`tests/integration/test_coalescing_neuron_split_attribution.py` (an end-to-end real-model
LIF run **and** the genuine scrambled-id collision). **But the PRODUCTION NF↔SCM gate
asserts identity-mapping-only** (`assert len(core_placements) == 1`,
`split_group_id is None`) and runs against a freshly-built identity mapping
(`build_identity_mapping_for_pipeline`) — it never reassembles the deployed
coalesced/output-tiled FRAGMENT mapping. So the fragment attribution path is **NOT
exercised in deployment**; only the test-only harness exercises it.

GAP-1 therefore **stays** `AttributionFidelity.VALUE_DOMAIN_ONLY`
(`KNOWN_CRACKED_REGIONS` in `coverage_ledger.py`): production per-neuron attribution under
coalescing+output-tiling is not gated. The collapse claims **only the deployed VALUE** for
`coalesced`; it does **not** over-claim per-neuron attribution. Closing GAP-1 would require
a production gate that reassembles the deployed packed/tiled mapping (the harness
reassembler is correct and ready) rather than rebuilding an identity mapping.

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
