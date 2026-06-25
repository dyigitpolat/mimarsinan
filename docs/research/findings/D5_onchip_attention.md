# D5 — the transformer frontier: on-chip attention + LayerNorm

**Status:** isolated research branch `wave7/d5-attention`. Ships ONE tested,
gated-by-isolation on-chip component (LayerNorm mean-centering, bit-exact /
float32-bounded) plus a COMPLETE, executable characterization of exactly which
attention/LN sub-ops are NOT chip-mappable and why. A genuine result in the
spirit of the residual Tier-1 `1/T` finding: a precise decomposition into a
realizable affine part and an intrinsically non-affine remainder.

Module: `src/mimarsinan/mapping/onchip_attention/`
(`onchip_layernorm.py`, `attention_mappability.py`, `test_onchip_attention.py`).

---

## 1. The chip primitive (the thing everything must reduce to)

The only genuine on-chip compute is a single `NeuralCore` crossbar
(`mapping/ir/types.py::NeuralCore.execute`):

```
y = clamp(relu(W @ x + b), 0, theta)        # W, b STATIC at map time
```

A composition of these realizes **exactly** the *piecewise-linear,
fixed-weight, affine* maps of the chip input. Everything else
(`nn.LayerNorm`, `nn.MultiheadAttention`, `F.softmax`, residual add, mean-pool)
currently lands as a **host-side `ComputeOp`** — deployed, but run OFF the
crossbar in host float (`docs/.../WS2_op_support_and_keystone.md`, Gaps G5–G7).
So "on-chip" for a transformer sub-op means: *expressible as static-weight
`clamp(relu(W x + b))`.* E7 already foreclosed the cheap host path; D5 asks the
hard question directly.

---

## 2. The decomposition (each verdict is backed by an executable proof)

`attention_mappability.affine_realizability_report(op)` returns a frozen
verdict per sub-op. The verdicts are not assertions of taste — each is proven
in `test_onchip_attention.py`:

| sub-op | on-chip? | reason_code | proof in the test suite |
|--------|----------|-------------|--------------------------|
| `qk_score` (QK^T) | **NO** | `bilinear_data_dependent` | cross-Hessian `d²(q·k)/dq dk = I` is nonzero ⇒ not affine in `[Q;K]` |
| `softmax` | **NO** | `nonlinear_exp_normalize` | Jacobian `diag(s)−ssᵀ` is input-dependent (differs at `z` vs `3z`); exp ∉ piecewise-linear set |
| `attention_value_matmul` (P·V) | **NO** | `data_dependent_weights` | the "weights" P are an activation, not a static parameter — same obstruction as QK^T |
| `layernorm_variance` (÷σ) | **NO** | `scale_invariant_division` | `LN(a·x)=LN(x)` for `a>0`; a linear `M` obeys `M(ax)=a·M(x)` ⇒ contradiction for `a≠1` |
| `layernorm_centering` (x−μ) | **YES** | `fixed_linear_projection` | `x−mean(x)=(I−(1/N)11ᵀ)x`, a fixed matrix; bit-exact to float64 epsilon |

`test_report_covers_every_attention_subop` asserts this table is **total** and
that the realizable set is **exactly** `{layernorm_centering}`.

### Why the three hard parts are intrinsically off-crossbar

- **QK^T / P·V — data-dependent matmul.** A crossbar multiplies the input by a
  matrix that is *frozen at map time*. Attention multiplies two *activations*
  (Q·Kᵀ, then P·V). The score is bilinear (nonzero cross-Hessian), so it is not
  an affine function of any fixed concatenation of the inputs. No static
  crossbar — and no composition of static crossbars — computes it. This is the
  same class as a "weights = activations" obstruction; it cannot be tiled away.
- **softmax — exp + data-dependent normalize.** The primitive is
  *piecewise-linear*; its Jacobian is constant inside a region. softmax's
  Jacobian varies continuously with the input. Neither the `exp` nor the
  divide-by-row-sum is in the `clamp(relu(affine))` set.
- **LayerNorm variance — scale-invariant division.** Dividing by `sqrt(var+ε)`
  makes LN scale-invariant. Scale invariance is provably incompatible with
  linearity, so the reciprocal-std is not ANY fixed matrix (it is a
  per-token data-dependent scalar divide, the same family as the softmax
  normalizer).

---

## 3. The realizable component (shipped, tested, on-chip)

`OnchipLayerNormCentering(num_features=N)` maps LayerNorm's mean-subtraction to
a **two-rail on-chip `NeuralCore` pair** and emits an `IRGraph` with **no host
`ComputeOp`** (`test_onchip_centering_emits_only_neural_cores`).

Mean-centering is the fixed projection `C = I − (1/N)·11ᵀ`
(`build_centering_matrix`; `C @ x == x − mean(x)`, verified to float64 epsilon;
`C @ C == C` — it is idempotent, a true projection). Because the crossbar
applies a ReLU, a *signed* projection is realized as two rails:

```
pos = clamp(relu( C x), 0, theta)      # the (C x)+ half
neg = clamp(relu(-C x), 0, theta)      # the (C x)- half
C x = pos - neg                        # param-free two-rail merge
```

Both rails are genuine on-chip cores; the `pos − neg` recombination is the same
param-free-merge family as the residual Tier-1 merge.

### The lock (bit-exact-or-bounded, honestly scoped)

The `NeuralCore` executor runs the crossbar in **float32** by construction. So
the lock is, exactly:

1. **Deterministic / bit-exact run-to-run** —
   `test_onchip_centering_is_deterministic_bit_exact` (`torch.equal` across two
   executions; no nondeterminism / data race).
2. **Bounded vs the mathematical reference** — the on-chip result matches
   `x − mean(x)` to **≤ one float32 epsilon (1.2e-6)**
   (`test_onchip_centering_bounded_to_math_reference`; measured max abs diff
   ≈ 7.9e-8). This is a *bounded* lock in the residual-Tier-1 spirit: a valid
   on-chip op with a precisely-characterized numeric gap (here, the chip's own
   float32 arithmetic), not a host-float-identical one.
3. **Characterized clamp bound** — the lock is exact only while the centered
   magnitude stays ≤ `clamp_ceiling`; the mapper exposes `exact_input_bound()`
   and a test confirms a super-ceiling input saturates the rail (a chip
   reality, surfaced, not hidden).

What is NOT shipped (and why): the ÷σ that would complete a full on-chip
LayerNorm is the `scale_invariant_division` row above — not chip-mappable. A
full deployment must either keep ÷σ as a (small) host op or fold a *learned,
data-independent* per-channel scale into the consumer Linear (a different,
approximate op, out of scope for a bit-exact lock).

---

## 4. The honest DoD-3 verdict

DoD-3 allows EITHER an on-chip attention/LN component OR an honestly-scoped
conv headline. D5 delivers the **former, scoped to its realizable core**, plus
the **complete negative characterization** of the rest:

- **Attention is intrinsically host-only.** QK^T, softmax, and P·V are *each*
  non-affine / data-dependent. There is no static-weight crossbar mapping for
  the attention score path, and none can be constructed by tiling or
  composition. A "linear-attention" variant does NOT rescue this: dropping the
  softmax still leaves the data-dependent QK^T / P·V products. The only on-chip
  attention is the degenerate *fixed-attention* case (attention weights frozen
  into parameters), which collapses to a plain on-chip Linear and is no longer
  attention.
- **LayerNorm splits cleanly.** Mean-centering IS on-chip (shipped, bit-exact /
  float32-bounded); variance normalization is NOT (scale-invariant division).
- **Conv headline stands unchanged.** The clean fully-on-chip vehicle remains
  the `{Linear, Conv1d, Conv2d}` + folded-BN + ReLU subset (the `mlp_mixer_core`
  / deep_cnn cascade); a ViT deploys with q/k/v/o + MLP Linears on-chip and the
  attention scoring + LN + softmax on host (WS2 §"Net"). D5 does not change
  that deployment; it characterizes precisely *why* the attention block cannot
  move on-chip and ships the one LN sub-part that can.

This is a genuine future-direction boundary, not a bug: **a transformer's
attention block is not reducible to the chip's static-weight piecewise-linear
primitive**, and the largest realizable on-chip LN piece is the mean-centering
projection.

---

## 5. How to reproduce

```bash
source env/bin/activate
PYTHONPATH="$PWD/src:$PYTHONPATH" python -m pytest \
  src/mimarsinan/mapping/onchip_attention/test_onchip_attention.py -q
# 13 passed
```
