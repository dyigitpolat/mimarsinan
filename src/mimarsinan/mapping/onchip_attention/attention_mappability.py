"""Formal verdicts on which attention/LayerNorm sub-ops fit the chip crossbar primitive."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffineRealizability:
    """Verdict on whether a sub-op maps to a static-weight crossbar."""

    op: str
    realizable: bool
    reason_code: str
    explanation: str

    KNOWN_OPS = (
        "qk_score",
        "softmax",
        "attention_value_matmul",
        "layernorm_variance",
        "layernorm_centering",
    )


_VERDICTS = {
    "qk_score": AffineRealizability(
        op="qk_score",
        realizable=False,
        reason_code="bilinear_data_dependent",
        explanation=(
            "The score S_ij = q_i . k_j is bilinear in the activation pair "
            "(Q, K): its cross second derivative d^2 S / dq dk is the identity "
            "(nonzero), so S is NOT an affine function of the concatenated "
            "[Q; K] input.  A crossbar W is static, so it can compute W [Q; K] "
            "but never the data-dependent product Q K^T."
        ),
    ),
    "softmax": AffineRealizability(
        op="softmax",
        realizable=False,
        reason_code="nonlinear_exp_normalize",
        explanation=(
            "softmax(z) = exp(z) / sum(exp(z)) needs an elementwise exp and a "
            "division by a per-row data-dependent normalizer.  The crossbar "
            "primitive clamp(relu(W x + b)) is piecewise-LINEAR; its Jacobian "
            "is constant within a region, whereas softmax's Jacobian "
            "diag(s) - s s^T varies continuously with the input.  Neither exp "
            "nor data-dependent division is in the primitive set."
        ),
    ),
    "attention_value_matmul": AffineRealizability(
        op="attention_value_matmul",
        realizable=False,
        reason_code="data_dependent_weights",
        explanation=(
            "The context P V uses the softmax probability matrix P as the "
            "'weights' multiplying V.  P is an activation (it depends on Q, K), "
            "not a static parameter, so this is a data-dependent matmul -- the "
            "same obstruction as QK^T.  A crossbar's weights are fixed at "
            "map time."
        ),
    ),
    "layernorm_variance": AffineRealizability(
        op="layernorm_variance",
        realizable=False,
        reason_code="scale_invariant_division",
        explanation=(
            "Dividing by sqrt(var + eps) makes LayerNorm scale-invariant: "
            "LN(a x) = LN(x) for a > 0.  A linear map M obeys M(a x) = a M(x); "
            "scale invariance contradicts linearity for any a != 1, so the "
            "variance normalization cannot be ANY fixed matrix.  The per-token "
            "reciprocal-std is a data-dependent scalar division, the same class "
            "as softmax's normalizer."
        ),
    ),
    "layernorm_centering": AffineRealizability(
        op="layernorm_centering",
        realizable=True,
        reason_code="fixed_linear_projection",
        explanation=(
            "Mean-subtraction x - mean(x) = (I - (1/N) 11^T) x is a FIXED "
            "linear projection with a data-independent matrix C.  It maps to "
            "on-chip NeuralCore crossbars via a signed two-rail ReLU "
            "decomposition (relu(C x) on a positive rail, relu(-C x) on a "
            "negative rail; the host/merge subtracts), bit-exact up to the "
            "rail clamp ceiling.  This is the realizable LayerNorm sub-part."
        ),
    ),
}


#: The two products of activations become realizable once the chip declares an operand core type (N2): one
#: operand is written into the core as its matrix once per inference, the other streamed through it.
_DYNAMIC_OPERAND_OPS = ("qk_score", "attention_value_matmul")


def affine_realizability_report(op: str, *, operand_core_type: bool = False) -> AffineRealizability:
    """Return the realizability verdict for ``op`` (raises ``KeyError`` if unknown); with a declared operand core
    type (N2), the two products of activations are realizable on dynamic operand cores."""
    if op not in _VERDICTS:
        raise KeyError(
            f"Unknown attention/LN sub-op {op!r}; "
            f"known ops: {sorted(AffineRealizability.KNOWN_OPS)}"
        )
    if operand_core_type and op in _DYNAMIC_OPERAND_OPS:
        static = _VERDICTS[op]
        return AffineRealizability(
            op=op, realizable=True, reason_code="dynamic_operand_core",
            explanation=(f"{static.explanation}  With an operand core type declared, one operand is written into "
                         "the core as its matrix once per inference (a payload on its producer's grid) and the "
                         "other streamed through it, so the product runs on cores (mapping/support/operand_lowering)."),
        )
    return _VERDICTS[op]
