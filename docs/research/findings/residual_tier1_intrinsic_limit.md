# Residual Tier-1 (on-chip param-free merge) — the intrinsic 1/T limit

**Status:** isolated research branch `wave4/residual-t1` (NOT merged — the headline bit-exactness goal
is intrinsically unreachable; honest xfails kept). A genuine *negative result* with a precise decomposition.

## The goal and why it does not bit-exactly reduce to Tier 0

Residual **Tier-0** (`DeepMLP(residual=True)`) adds the two streams **host-side** and is bit-exact.
**Tier-1** attempts the same add as an **on-chip param-free merge** (two spike streams summed at a merge
core / segment boundary). The Wave-4 round-3 attempt instrumented the ~`1/T` residual gap and decomposed
it into two independent components:

### Component A — CLOSEABLE (a shared-HCM-fill alignment)
A skip-window truncation in the **shared HCM input fill** (`stage_io` /
`_fill_signal_tensor_from_spans`, `lif_step`): the raw-input skip is not delayed to the merge core's
latency window. Closing it drives **NF == HCM to float64 `atol=0` over 8 seeds**. This is a real
alignment fix worth doing on its own (it tightens the NF↔HCM residual-merge fidelity lock) — but the
code lives outside the residual unit's ownership (the shared HCM fill is parity-critical; every
torch↔sim fidelity lock depends on it), so it belongs in a dedicated, carefully-verified unit.

### Component B — INTRINSIC (an in-segment IF-head re-quantization)
After Component A is closed, an **irreducible** difference remains: the in-segment IF head
**re-quantizes** the merged spike train differently than the host-add path's uniform re-encoding, so
the on-chip merge `≠` host-add by **exactly 1 spike (`0.125` at `T=8`, i.e. `1/T`) by construction**.
Re-uniformizing the merged stream on-chip requires a **host round-trip — which is exactly Tier 0**.

## The honest conclusion

An **in-segment** on-chip param-free residual merge **cannot be bit-exact to the Tier-0 host-add
reference** under single-spike/IF semantics; it is a *different, valid* deployment that differs by a
characterized `1/T` re-quantization. The two real paths forward are therefore:

1. **Redefine the Tier-1 success criterion** away from "bit-exact to host-add" to "a valid on-chip
   merge with a characterized, bounded `1/T` re-quantization" (then certify the small accuracy effect
   empirically) — this is a *capability* the toolchain can support, not a bug to fix.
2. **Close Component A** in a dedicated shared-HCM-fill unit (NF==HCM `atol=0`), which tightens the
   reference even though it does not remove the intrinsic Component-B gap.

This is recorded as a **genuine future-direction gap** (§5 of `RESEARCH_DOCUMENT.md`): Tier-1 on-chip
residual merge is feasible as a `1/T`-characterized deployment, not as a host-add-identical one.
