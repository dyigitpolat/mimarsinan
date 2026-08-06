# SCM/HCM action plan — kill the regression, then the remaining resource defects

**Date:** 2026-08-07 · **Base:** `b1ed9a6a` (Units 1–4 landed) · all numbers below
are MEASURED on the real 4,925-core ViT unless marked *projected*.

---

## 1. Where things stand (p11, sealed proof)

| defect | before | after Units 1–4 | verdict |
|---|---|---|---|
| SCM ir_graph pickle | 45.86 GB | 0.83 GB | fixed (55×) |
| HCM hybrid pickle | 35.36 GB | 1.33 GB | fixed (27×) |
| prune RSS | +42 GB | +0.0 MiB | fixed |
| parity-gate RSS | +113.3 GB | −1.5 GB | fixed |
| parity-gate wall | 421 s | 263 s | improved |
| SCM step | 12,114 s | 1,281 s | 9.5× (holding) |
| **HCM step** | **1,742 s** | **2,520 s** | **REGRESSED +778 s** |
| **GUI resource sources on disk** | **179 GB** | **179 GB** | **pre-existing, untouched, now the largest defect** |

Correctness held throughout: `softcore_elimination.json` sha256-identical,
metric 0.9699 Δ=+0.0000 at SCM, quantization verification, and the on-chip
simulation test.

## 2. The regression, attributed (not guessed)

Event timestamps split HCM at the `weight_programming` event:

| phase | old code | new code | delta |
|---|---|---|---|
| build + pack + pickle entry | 161 s | **92 s** | −69 s (descriptors pack faster) |
| everything after (sim test, snapshot, persist, save_cache) | 1,581 s | **2,428 s** | **+847 s** |

Measured causes inside that segment:

- **The snapshot walk forces a full materialization sweep**: 4,925 calls to
  `get_core_matrix()`, **92.9 GB churned, 49.6 s** (old code read ready dense
  grids). Every call returns a FRESH object (`a is b` → False), so nothing
  downstream can memo on identity.
- **The upload memo is keyed by `id(ndarray)`** (`value_execution._upload`),
  so fresh materializations never hit: every stage of every forward
  re-materializes AND re-uploads. With 75 neural stages this is the dominant
  term of the remaining ~800 s.

**The decisive fact that makes the fix cheap:** those 4,925 materializations
represent only **27 distinct payloads**. `core_matrix_key()` — already built by
Unit 2 — returns 27 distinct keys over 4,925 cores; content hashing confirms
it (600 cores → 5 payloads, multiplicities 197/197/99/97/10 = the ViT's token
structure). 27 × 18.9 MB = **510 MB** holds every distinct grid in the program.

## 3. Unit 5 — content-keyed materialization + upload memo (kills the regression)

**Design.** One memo, keyed by `core_matrix_key()` (stable content identity,
NOT object identity), serving both seams:

- `HardCore.get_core_matrix()` consults a process-level bounded memo before
  materializing; a hit returns the shared array.
- `value_execution._upload` keys its device-tensor cache by the same key, so
  one device tensor per distinct payload per device.

**Safety.** Cached arrays are handed out with `flags.writeable = False`: any
consumer that tries to mutate a shared payload fails LOUD instead of silently
corrupting its siblings (build-time writers assign `core_matrix` directly and
never touch the memo). Bound the memo by a configured budget (default ~2 GB,
LRU); when eviction occurs, `log()` it — no silent caps.

**Why it is bit-exact.** The memo returns the array the materializer would
have produced for that key; equality of key ⇒ equality of every input to the
materialization recipe (bank payload, keep-index arrays, offsets, padded
shape, dtype). Gate: extend the existing frozen-oracle differential to assert
`hex(memoized) == hex(freshly materialized)` for every core of every vehicle,
plus a mutation test (perturb one cached payload → differential must fail).

**Expected** *(projected)*: snapshot walk 49.6 s → ~1 s; sim test loses its
per-stage re-materialize/re-upload; HCM **2,520 s → ~1,400–1,600 s**, i.e.
below the old 1,742 s rather than merely back to it. Peak weight residency
~510 MB instead of 93 GB of churn.

## 4. Unit 6 — payload-identity resource persistence (kills 179 GB)

**Evidence.** Every run writes **179 GB** of GUI resource sources (95 GB SCM +
84 GB HCM), and this is byte-identical between old and new code — a
pre-existing defect, unrelated to Units 1–4, and now larger than everything
they fixed. Cause: `save_resource_source` writes each core's heatmap as an
independent dense array file, so the 27 distinct payloads are written 4,925
times (plus the IR side: per-core heatmaps, pre-pruning views, connectivity).

**Design (the Unit-1 pattern, one level out).** Persist by payload identity:
write each distinct payload once under a content-addressed name, and store
per-core sources as small records `(payload ref, row/col window, mask refs)`.
Rendering resolves the reference at render time — the renderer already
receives a source object, so this is a source-type change, not a UI change.

**Expected** *(projected)*: 179 GB → **~1–3 GB**; the write time inside SCM
and HCM disappears with it. Also removes a real operational hazard: at 179 GB
per run the 21 TB volume (12 TB used) tolerates only ~50 more runs.

## 5. Unit 7 — verifier findings (the panel's own list)

Two of three adversarial panels returned `pass: false`. Their non-minor
findings, all confirmed by re-derivation:

| sev | finding | action |
|---|---|---|
| blocker | snapshot walker materializes a fresh 18.9 MB f64 grid per descriptor core | subsumed by Units 5 + 6 |
| major | `HardCore.__setstate__` never defaults `matrix_placements`, so pre-`f2cf5405` pickles raise on first access | `state.setdefault(...)` + a legacy-state regression test |
| major | `HardCoreMapping.output_sources` still pickles as SpikeSource object soup (27.7 B/source) | extend Unit 4's `__getstate__` seam to `HardCoreMapping` |
| major | Unit 3's residency goldens survive REVERSING the per-ordinal weight list (mutation-check passed the whole suite) | add a ≥2-distinct-bank resident-chain vehicle with heterogeneous weights AND biases; pin float-hex goldens |
| minor ×5 | ARCHITECTURE.md drift; over-counted keep row in a storage bound; IR wiring still object soup; fragment path materializes; residency law can't see content identity | fold into the units below / fix in place |

The residency mutation gap is the one that matters most for trust: a golden
that cannot fail is not a gate. It gets fixed before anything else in Unit 7.

## 6. Unit 8 — the SCM remainder (851 s analysis → ~350–400 s)

Measured hierarchy inside the 851 s elimination ledger: **4–5 × ~110–135 s of
per-pass context setup ≈ 500 s** (graph indexes, consumer maps, value-based
seeding rebuilt identically per arm and for the replay), then ~16 s core
gathers, ~10 s input assembly, ~130 s emit.

**Design.** Build the immutable indexes ONCE per run and thread them through
arms + replay (the probe-memo pattern from the last arc, applied to structure
instead of probes): split `GlobalPruningContext` into an immutable
`GraphIndex` (consumer axons, transfer index, output markers, bank lookups)
and per-arm mutable state (masks, lattice). Gate: golden fingerprint +
arms/ledger tests + the real-vehicle record hash.

**Expected** *(projected)*: analysis 851 s → **~350–400 s**; SCM step
1,281 s → ~800 s.

## 7. Unit 9 — IR wiring compression (0.83 GB → ~0.1 GB)

The SCM pickle's remaining 0.83 GB is dominated by ~9.4M `IRSource` objects in
`NeuralCore.input_sources`. Apply Unit 4's range-compression seam at the IR
boundary (an `IRSource` spans encoder mirroring `spike_source_spans`, or
`__getstate__` on the graph). Gate: round-trip field equality + a storage
regression test + the elimination record hash.

## 8. Sequencing, gates, and how each is proven

1. **U7a residency mutation gap** (trust first — a gate that can't fail).
2. **U5 content-keyed memo** — the regression killer; re-measure HCM alone.
3. **U6 payload-identity persistence** — the 179 GB; re-measure both steps.
4. **U7b/c** legacy-state default + `output_sources` seam + minors.
5. **U8 one context build per run** — re-measure SCM.
6. **U9 IR wiring** — re-measure the pickle.

Every unit: tests FIRST; full suite + typecheck + LOC/sibling ratchets green
before commit; no assertion weakened. Bit-identity is proven per unit by the
frozen-oracle differentials already in place, and the whole stack is re-proven
end-to-end by a final run whose acceptance is unchanged from p11's:
`softcore_elimination.json` sha256-identical, metric 0.9699 Δ=+0.0000, on-chip
simulation test 0.9699.

**Projected end state** *(to be replaced by measurement)*: SCM ~800 s, HCM
~1,400 s, artifacts ~1.4 GB, GUI sources ~2 GB, peak RSS deltas bounded —
against the arc's starting point of 17,348 s SCM, 1,742 s HCM, 81 GB of
artifacts and 179 GB of sources.

## 9. What this plan does NOT do

- No change to any deployment semantics, in any mode (the memo is a cache; the
  persistence change is a storage format; the context split is structural).
- No touching of the LIF per-hop fusion work (separate, `docs/lif_hop_fused_mapping_design.md`).
- No new speculative optimization: every unit above is anchored to a measured
  number in §1–§2, and each ships behind a gate that can fail.
