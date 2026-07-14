"""[ttfs_exact_qat] TTFS-quantized exact-QAT arm: predicate + per-model marker.

The TTFS analog of ``lif_exact_qat``: the AQ stage trains the exact deployed
TTFS ceil staircase with theta in-loop (``TTFSCountStaircaseDecorator``),
replacing the float shift + floor-quantize proxy. Unlike LIF, TTFS is
analytical, so there is NO per-hop re-timing twin — the arm is theta promotion
+ the exact-kernel decorator. Knob off is byte-identical.
"""

from __future__ import annotations

_TTFSQ_EXACT_QAT_ATTR = "_mbh_ttfsq_exact_qat"


def ttfsq_exact_qat_active(pipeline_config) -> bool:
    """The ``ttfsq_exact_qat`` knob is on AND the mode is ttfs_quantized.

    Non-ttfsq modes with the knob armed fail LOUD (silent disarming would be a
    silent Goodhart hole — the arm changes the trained object)."""
    if not bool(pipeline_config.get("ttfsq_exact_qat", False)):
        return False
    mode = str(pipeline_config.get("spiking_mode", "lif"))
    if mode != "ttfs_quantized":
        raise ValueError(
            f"ttfsq_exact_qat is only meaningful for spiking_mode="
            f"'ttfs_quantized'; got {mode!r}. Remove the key."
        )
    return True


def mark_ttfsq_exact_qat(perceptron) -> None:
    """Persistently mark ``perceptron`` as trained through the exact TTFS ceil staircase."""
    setattr(perceptron, _TTFSQ_EXACT_QAT_ATTR, True)


def model_trained_ttfsq_exact(model) -> bool:
    """Whether the model's AQ endpoint was the exact TTFS staircase (all-or-none)."""
    flags = [
        bool(getattr(p, _TTFSQ_EXACT_QAT_ATTR, False)) for p in model.get_perceptrons()
    ]
    if not flags or not any(flags):
        return False
    assert all(flags), (
        "ttfsq-exact QAT marker is inconsistent: "
        f"{sum(flags)}/{len(flags)} perceptrons are marked; the exact-staircase "
        "endpoint must cover every perceptron or none."
    )
    return True
