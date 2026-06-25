# mapping/onchip_attention/ — Transformer-frontier op→core mappability (D5)

Answers, with executable proofs, which self-attention / LayerNorm sub-ops fit
the chip's static-weight `clamp(relu(W x + b))` crossbar primitive — and maps
the one realizable piece (LayerNorm mean-centering) to on-chip `NeuralCore`s.

Research finding: `docs/research/findings/D5_onchip_attention.md`.

## Files

| File | Exports | Role |
|------|---------|------|
| `attention_mappability.py` | `AffineRealizability`, `affine_realizability_report` | Per-sub-op realizability verdicts. `affine_realizability_report(op)` returns a frozen `AffineRealizability(realizable, reason_code, explanation)` for each of `AffineRealizability.KNOWN_OPS` (`qk_score`, `softmax`, `attention_value_matmul`, `layernorm_variance`, `layernorm_centering`). Only `layernorm_centering` is realizable; the rest are bilinear/data-dependent (QK^T, P·V), exp/normalize (softmax), or scale-invariant division (LN variance). |
| `onchip_layernorm.py` | `OnchipLayerNormCentering`, `build_centering_matrix` | Maps LayerNorm mean-subtraction `x−mean(x) = (I−(1/N)11ᵀ)x` to a signed **two-rail** on-chip `NeuralCore` pair (positive `relu(Cx)`, negative `relu(−Cx)`; param-free `pos−neg` merge). `to_ir_graph()` emits an `IRGraph` with two NeuralCores and NO host `ComputeOp`. `reconstruct_centered(buffers)` does the rail merge; `exact_input_bound()` exposes the `clamp_ceiling` above which a rail saturates. |
| `test_onchip_attention.py` | — | TESTS-FIRST. Bit-exact / float32-bounded lock for the centering rails; executable proofs of the non-mappability verdicts (cross-Hessian, softmax Jacobian, LN scale-invariance); totality of the verdict table. |

## Dependencies

- **Internal**: `mapping.ir` (`IRGraph`, `IRSource`, `NeuralCore`).
- **External**: `numpy`, `torch`.

## Dependents

- None in the deployment path (isolated research module; default pipeline
  behavior is byte-identical). The on-chip surface for production deployment is
  still `{Linear, Conv1d, Conv2d}` + folded BN + ReLU — attention scoring,
  softmax, and LN variance remain host `ComputeOp`s by construction (D5 proves
  why they cannot move on-chip).

## Invariants

- The crossbar primitive is `clamp(relu(W x + b), 0, theta)` with STATIC W, b —
  realizes exactly the piecewise-linear fixed-weight affine maps.
- A signed projection needs TWO rails on a ReLU crossbar; `C x = pos − neg`.
- Mean-centering is idempotent (`C @ C == C`): a true on-chip projection.
