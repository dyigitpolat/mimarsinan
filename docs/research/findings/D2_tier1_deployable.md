# D2 — Residual Tier-1 as a VALID, characterized on-chip deployment

**Status:** DELIVERED on `wave9/d2-tier1-deployable`. The Tier-1 on-chip param-free
residual merge is wired as a **selectable deployment mode** (`IRMapping(onchip_residual_merge=True)`,
default off → Tier-0 host-add, byte-identical) and the bounded `~1/T` effect is
**MEASURED** on a small residual model — confirming a CHARACTERIZED, bounded
re-quantization, NOT a blow-up.

This realizes path #1 of `residual_tier1_intrinsic_limit.md`: redefine the Tier-1
success criterion away from "bit-exact to host-add" to "a valid on-chip merge with a
characterized, bounded `1/T` re-quant", then certify the small accuracy effect
empirically.

## What was wired (the selectable deployment mode)

`mapping/support/residual_merge.py::lower_residual_adds_to_onchip_merge` rewrites a
param-free equal-width residual add `y = z + F(z)` (a host `ComputeAdapter(operator.add)`,
Tier 0) into a frozen identity-concat merge `Perceptron` (`[I | I]` weight, no bias,
signed-IF `Identity` activation) fed by a `_ResidualConcatMapper`. The sum then runs
on the crossbar (`W @ [z ; F] = z + F`), param-free, on-chip, in a SINGLE neural
segment. Unequal-width (projection) residuals are left as host ComputeOps.

The pass is invoked from `LayoutIRMapping.map` only when the new `onchip_residual_merge`
config flag is set. The flag threads through `IRMappingCore.__init__` →
`LayoutIRMapping` (dataclass field, default `False`). **Default off** leaves the host
ComputeOp add untouched, so every existing `IRMapping(...)` construction site
(`soft_core_mapping_step`, the verifier, search, the pretrained bridge, etc.) is
byte-identical to the pre-Tier-1 path.

## The measurement (deployed Tier-1 vs deployed Tier-0)

Small residual model: `stem(16→24) → ReLU → F: Linear(24→24) → ReLU → skip-add →
head(24→10)`, LIF deployment, identity mapping config, packed HCM through the
production config gate. Both Tier-0 (host add) and Tier-1 (on-chip merge) are real
deployed `SpikingHybridCoreFlow` builds of the same weights; we compare their
per-output spike counts (range `[0, T]`).

**T = 8, 8 seeds × 16 samples × 10 outputs = 1280 deployed output values:**

| metric | value |
|--------|-------|
| differing outputs | **40 / 1280 = 3.12 %** |
| max \|Δcount\| | **1 spike** |
| max \|Δrate\| (= count / T) | **0.125 = 1/T** |
| mean \|Δcount\| (per output) | ~0.03 spikes |

**The `1/T` law across T (8 seeds each, worst-case over seeds):**

| T | max \|Δcount\| | max \|Δrate\| | 1/T |
|---|----------------|----------------|------|
| 4  | 1 spike | 0.2500 | 0.2500 |
| 8  | 1 spike | 0.1250 | 0.1250 |
| 16 | 1 spike | 0.0625 | 0.0625 |
| 32 | 1 spike | 0.0313 | 0.0313 |

The absolute COUNT delta stays a **fixed 1 spike** as `T` grows, so the VALUE (rate)
delta is **exactly `1/T`** and SHRINKS with `T`. This is the decisive signature of a
bounded re-quantization: a genuine cascade misalignment would instead drift the count
toward full scale (`T` spikes) and leave the rate delta roughly constant in `T`.

## Why exactly `1/T` (the intrinsic re-quant, Component B)

The on-chip merge is a spiking integrate-and-fire neuron: its output is a spike
*train* whose timing is the IF-sum of the two input trains, not a uniform encoding of
the rate `(z + F)/T`. The in-segment head integrates that raw train directly, whereas
the Tier-0 host-add path crosses a segment boundary that decodes the merge to a rate
and re-encodes it uniformly. An IF head with mixed-sign weights re-quantizes a
non-uniform train differently from a uniform one — the membrane leftover at the window
end differs by at most one spike. Re-uniformizing the merge train on-chip needs a host
round-trip, which is exactly Tier 0. So `on-chip == host-add` is unreachable *by
construction* for an in-segment spiking merge; the difference is a bounded one-spike
(`1/T`) re-quant, certified here as small (3 % of outputs, never > 1 spike).

(Component A — the shared-HCM-fill skip-window truncation that would tighten `NF==HCM`
to float64 `atol=0` — lives in shared parity-critical files outside this unit's
ownership and is not required for the bound: the measured worst case here already sits
at the 1-spike Component-B floor, not the 2-spike `A + B` ceiling.)

## Tests (tests-first, real measurements, no fakes)

- `tests/unit/mapping/test_residual_merge_lowering.py` — the lowering pass directly
  (equal-width add → frozen `[I | I]` merge Perceptron, param-free, no bias, Identity
  activation; no-residual graph untouched; unequal-width left host) **and** the
  IRMapping config gate (off keeps the host add; on adds exactly one merge NeuralCore).
- `tests/integration/test_residual_tier1_deployable.py` — the production config gate
  end-to-end: (1) default-off deploys byte-identical to the no-flag baseline; (2)
  flag-on deploys as a single on-chip neural segment with no host add and no extra host
  params; (3) the deployed Tier-1-vs-Tier-0 delta is **measured** within the `~1/T`
  re-quant bound across seeds, the `1/T`-scaling law is locked across `T ∈ {4,8,16,32}`,
  and an honesty lock confirms Tier-1 is a *different* valid deployment (not a silent
  re-derivation of the host round-trip).

## Verdict

Tier-1 residual on-chip merge is a **VALID, selectable, characterized deployment**: it
keeps a param-free residual on-chip (one segment, no host add, no host params) and
differs from the Tier-0 host-add reference by a **measured, bounded `1/T` (one-spike)
re-quant** that shrinks with `T` — not a blow-up. It is intentionally NOT the default
(Tier-0 host-add stays the byte-identical default); it is the on-chip-majority-friendly
option when keeping the residual on the crossbar matters more than exact host-add
parity.
