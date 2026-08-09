"""``quantize_ir_graph`` bits-sensitivity contract (BC-1 postmortem pins).

BC-1 measured BYTE-IDENTICAL elimination ledgers for 8-bit and 4-bit
quantization of the same conv IR (80/81 cores dead).  Root cause is a call
misuse, not a quantize-path defect:

- ``chip_quantize._matrix_scale`` treats ANY nonzero ``parameter_scale`` as the
  authoritative NAPQ-installed grid scale and returns it as-is; ``bits`` then
  only supplies the clip bounds ``[q_min, q_max]`` and the integer dtype.  The
  ``q_max / max|w|`` derivation runs ONLY for the ``parameter_scale == 0.0``
  sentinel (used e.g. by depth-balancing relays).
- A freshly mapped IR that never went through
  ``NormalizationAwarePerceptronQuantization`` carries the DATACLASS DEFAULT
  ``parameter_scale = torch.tensor(1.0)`` (``mapping/ir/types.py``), which is
  indistinguishable from an installed scale of exactly 1.0.  Quantizing such a
  graph computes ``clip(round(W * 1.0))`` at every width: typical conv floats
  (|w| < 0.5) all round to 0, the ledger kills almost every core, and since
  every rounded value already fits the 4-bit range (both widths cast to int8),
  the 8b and 4b outputs are byte-identical.

Correct standalone usage on a raw (NAPQ-less) IR: set ``parameter_scale`` to
the 0.0 sentinel on every bank and every bank-less core, so the scale is
derived per-matrix as ``q_max / max|w|`` — then ``bits`` genuinely changes the
matrices, thresholds, and downstream ledgers.  (The pipeline instead installs
per-perceptron scales via NAPQ at the same ``weight_bits`` before mapping.)
"""

import copy

import numpy as np
import pytest
import torch

from mimarsinan.mapping.export.chip_quantize import (
    quantize_ir_graph,
    verify_ir_graph_quantized,
)
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.ir.weight_bank import WeightBank
from mimarsinan.transformations.quantization_bounds import quantization_bounds

DERIVE_SCALE_SENTINEL = 0.0  # parameter_scale == 0.0 -> derive q_max / max|w|


def _rich_matrix() -> np.ndarray:
    """(8, 8) float weights with rich dynamic range, |w| <= 0.5, exact zeros."""
    w = np.linspace(-0.5, 0.5, 64, dtype=np.float64).reshape(8, 8)
    w[0, 0] = 0.0
    w[3, 5] = 0.0
    w[7, 7] = 0.0
    return w


def _two_core_graph(parameter_scale: float) -> IRGraph:
    """One bank-backed core + one owned-matrix core, same rich weights."""
    matrix = _rich_matrix()
    n_axons, n_neurons = matrix.shape
    scale = torch.tensor(float(parameter_scale))
    bank = WeightBank(id=0, core_matrix=matrix.copy(), parameter_scale=scale.clone())
    sources = np.array(
        [IRSource(node_id=-2, index=i) for i in range(n_axons)]
    )
    banked = NeuralCore(
        id=0,
        name="banked",
        input_sources=sources.copy(),
        weight_bank_id=0,
        weight_row_slice=(0, n_neurons),
        parameter_scale=scale.clone(),
    )
    owned = NeuralCore(
        id=1,
        name="owned",
        input_sources=sources.copy(),
        core_matrix=matrix.copy(),
        parameter_scale=scale.clone(),
    )
    out = np.array([IRSource(node_id=1, index=j) for j in range(n_neurons)])
    return IRGraph(nodes=[banked, owned], output_sources=out, weight_banks={0: bank})


def _quantize_at(graph: IRGraph, bits: int) -> IRGraph:
    g = copy.deepcopy(graph)
    quantize_ir_graph(g, bits, weight_quantization=True)
    return g


def _matrices(graph: IRGraph) -> dict[str, np.ndarray]:
    return {
        "bank": graph.weight_banks[0].core_matrix,
        "owned": graph.nodes[1].core_matrix,
    }


def test_bits_change_quantized_matrices_with_derived_scale():
    """With the 0.0 sentinel (scale derived from bits), 8b vs 4b MUST differ:
    max|q| saturates to each width's q_max (within the floor lattice step)
    and threshold == floor(q_max / max|w|) — the INTEGER theta register
    contract (nevresim threshold_t=int; fractional theta was the 2026-08-09
    count-divergence root cause)."""
    base = _two_core_graph(DERIVE_SCALE_SENTINEL)
    w_max = float(np.max(np.abs(_rich_matrix())))
    g8, g4 = _quantize_at(base, 8), _quantize_at(base, 4)

    for bits, g in ((8, g8), (4, g4)):
        _, q_max = quantization_bounds(bits)
        expected_scale = max(1.0, float(np.floor(q_max / w_max)))
        for name, mat in _matrices(g).items():
            assert np.issubdtype(mat.dtype, np.integer), (bits, name, mat.dtype)
            assert q_max - 1 <= int(np.max(np.abs(mat))) <= q_max, (
                f"{bits}b {name}: derived-scale quantization must saturate "
                f"max|q| to q_max={q_max} within the floor-lattice step, "
                f"got {int(np.max(np.abs(mat)))}"
            )
        for node in g.get_neural_cores():
            assert node.threshold == pytest.approx(expected_scale), (
                f"{bits}b {node.name}: threshold must be the integral derived "
                f"scale floor(q_max/max|w|) = {expected_scale}, got {node.threshold}"
            )
        verify_ir_graph_quantized(g, bits)

    for name in ("bank", "owned"):
        m8, m4 = _matrices(g8)[name], _matrices(g4)[name]
        assert not np.array_equal(m8, m4), (
            f"{name}: 8-bit and 4-bit quantization produced byte-identical "
            f"matrices — the bits argument was inert (BC-1 failure mode)"
        )

    # The 8b matrix uses codes outside the 4-bit range: the 4b verification
    # gate must reject it (bits-sensitivity visible to the verifier too).
    with pytest.raises(AssertionError):
        verify_ir_graph_quantized(g8, 4)


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "parameter_scale", [DERIVE_SCALE_SENTINEL, 10.0], ids=["derived", "installed"]
)
def test_exact_zeros_stay_exact_zeros_at_every_width(bits, parameter_scale):
    """Exact float zeros must quantize to exact integer zeros at every width,
    for both derived-scale and installed-scale paths."""
    base = _two_core_graph(parameter_scale)
    zero_positions = np.argwhere(_rich_matrix() == 0.0)
    assert len(zero_positions) == 3
    g = _quantize_at(base, bits)
    for name, mat in _matrices(g).items():
        for r, c in zero_positions:
            assert mat[r, c] == 0, (
                f"{bits}b {name}: exact float zero at ({r}, {c}) quantized to "
                f"{mat[r, c]} != 0"
            )


def test_installed_scale_wins_and_bits_only_clips():
    """Contract pin: a nonzero parameter_scale is the NAPQ-installed grid and
    is used verbatim; bits affects ONLY the clip bounds (and dtype)."""
    installed = 24.0  # 0.5 * 24 = 12: inside 8-bit range, outside 4-bit range
    base = _two_core_graph(installed)
    g8, g4 = _quantize_at(base, 8), _quantize_at(base, 4)

    w = _rich_matrix()
    for bits, g in ((8, g8), (4, g4)):
        q_min, q_max = quantization_bounds(bits)
        expected = np.clip(np.round(w * installed), q_min, q_max)
        for name, mat in _matrices(g).items():
            np.testing.assert_array_equal(
                mat.astype(np.float64), expected,
                err_msg=f"{bits}b {name}: installed-scale path must be "
                f"clip(round(W * installed_scale), q_min, q_max)",
            )
        for node in g.get_neural_cores():
            assert node.threshold == pytest.approx(installed)

    assert int(np.max(_matrices(g8)["owned"])) == 12
    assert int(np.max(_matrices(g4)["owned"])) == 7  # 4-bit clip engaged


def test_bc1_misuse_default_scale_makes_bits_inert():
    """BC-1 reproduction (executable documentation of the misuse): on a fresh
    IR the dataclass DEFAULT parameter_scale=1.0 masquerades as an installed
    grid, so sub-0.5 conv floats all round to zero and 8b vs 4b outputs are
    byte-identical — exactly the measured 'identical ledgers, 80/81 dead'.
    BC-1b must NOT call it this way: install NAPQ scales or set the 0.0
    sentinel first (see module docstring)."""
    base = _two_core_graph(parameter_scale=1.0)  # the types.py default
    w = _rich_matrix()
    assert float(np.max(np.abs(w))) < 0.5 + 1e-12
    g8, g4 = _quantize_at(base, 8), _quantize_at(base, 4)
    for name in ("bank", "owned"):
        m8, m4 = _matrices(g8)[name], _matrices(g4)[name]
        assert np.array_equal(m8, m4) and m8.dtype == m4.dtype, name
        assert not np.any(m8), (
            f"{name}: scale=1.0 on sub-0.5 floats rounds everything to zero"
        )
