"""The operand form of a Contraction (N2): its batch, m, k and n axes, and the two lowerings and their prices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

B_STATIONARY, AT_STATIONARY = "B", "A^T"
#: The census name of a Contraction the operand rule does not lower (it keeps G1's host op).
OPERAND_FORM_OUTSIDE = "Contraction form outside the operand rule"


@dataclass(frozen=True)
class OperandForm:
    """``C = A B`` per index of the batch axes: A is (batch..., m, k), B (batch..., k, n), C (batch..., m, n).
    ``a_perm`` and ``b_perm`` take each operand's per-sample axes to that order; ``out_perm`` takes the canonical
    C to the traced output's axis order."""

    batch: Tuple[int, ...]
    m: int
    k: int
    n: int
    a_perm: Tuple[int, ...]
    b_perm: Tuple[int, ...]
    out_perm: Tuple[int, ...]

    @property
    def banks(self) -> int:
        return int(math.prod(self.batch))


@dataclass(frozen=True)
class Lowering:
    """One of the two lowerings: B stationary (k x n written, m rows of A streamed) or A^T stationary
    (k x m written, n columns of B streamed, C^T read back through index wiring)."""

    form: OperandForm
    stationary: str

    @property
    def axons(self) -> int:
        return self.form.k

    @property
    def neurons(self) -> int:
        return self.form.n if self.stationary == B_STATIONARY else self.form.m

    @property
    def instances(self) -> int:
        return self.form.m if self.stationary == B_STATIONARY else self.form.n

    @property
    def cells(self) -> int:
        return self.form.banks * self.axons * self.neurons

    @property
    def streamed_elements(self) -> int:
        return self.form.banks * self.instances * self.form.k

    @property
    def output_elements(self) -> int:
        return self.form.banks * self.form.m * self.form.n


def _no_unpriced(_lowering: Lowering) -> Tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class LoweringPrice:
    """[U-J4 targeted] A lowering's energy per inference (J), called as the price, and the rows its price reads that
    the physics does not declare (``unpriced``), which ``stationary_operand`` names in its printed reason."""

    energy: Callable[[Lowering], float]
    unpriced: Callable[[Lowering], Tuple[str, ...]] = _no_unpriced

    def __call__(self, lowering: Lowering) -> float:
        return self.energy(lowering)


def _labels(term: str, rank: int) -> Optional[List[str]]:
    if "..." not in term:
        return list(term) if len(term) == rank else None
    head, tail = term.split("...")
    covered = rank - len(head) - len(tail)
    return None if covered < 0 else [*head, *(f".{i}" for i in range(covered)), *tail]


def operand_rule_form(contraction) -> Optional[OperandForm]:
    """The operand form of ``contraction`` (G1's interface), or None outside the rule: two operands, the sample
    axis leading, one summed axis, one kept axis per operand, every other axis a batch axis both share at one
    extent, no diagonal and no broadcast."""
    spec = str(contraction.einsum_spec).replace(" ", "")
    if "->" not in spec:
        return None
    inputs, out_term = spec.split("->")
    terms = inputs.split(",")
    shapes = [tuple(int(d) for d in s) for s in contraction.operand_shapes]
    if len(terms) != 2 or len(shapes) != 2:
        return None
    la, lb = _labels(terms[0], len(shapes[0]) + 1), _labels(terms[1], len(shapes[1]) + 1)
    if la is None or lb is None or ("..." in terms[0]) != ("..." in terms[1]):
        return None
    ellipsis = [label for label in la if label.startswith(".")]
    lo = _labels(out_term, len(out_term.replace("...", "")) + len(ellipsis))
    if lo is None or len({la[0], lb[0], lo[0]}) != 1:
        return None
    la, lb, lo = la[1:], lb[1:], lo[1:]
    if len(set(la)) != len(la) or len(set(lb)) != len(lb) or len(set(lo)) != len(lo):
        return None
    summed = (set(la) & set(lb)) - set(lo)
    m_axes, n_axes = set(la) - set(lb), set(lb) - set(la)
    batch = [label for label in lo if label in la and label in lb]
    if len(summed) != 1 or len(m_axes) != 1 or len(n_axes) != 1 or not m_axes | n_axes <= set(lo):
        return None
    if set(lo) != set(batch) | m_axes | n_axes:
        return None
    extent: Dict[str, int] = {}
    for labels, shape in ((la, shapes[0]), (lb, shapes[1])):
        for label, size in zip(labels, shape):
            if extent.setdefault(label, size) != size:
                return None
    (k,), (m,), (n,) = summed, m_axes, n_axes
    canonical = [*batch, m, n]
    return OperandForm(
        batch=tuple(extent[b] for b in batch), m=extent[m], k=extent[k], n=extent[n],
        a_perm=tuple(la.index(x) for x in (*batch, m, k)), b_perm=tuple(lb.index(x) for x in (*batch, k, n)),
        out_perm=tuple(canonical.index(x) for x in lo),
    )


def stationary_operand(form: OperandForm, price: Callable[[Lowering], float],
                       cost_only: bool, why: str = "the role decides which operand takes the payload grid"
                       ) -> Tuple[Lowering, str]:
    """(the lowering, the reason printed): the cheaper of the two where the choice is cost-only (ties to fewer
    bank cells, then to B stationary); B stationary, the declared default, where the choice bears on the function
    (``why`` says how). [U-J4 targeted] A ``LoweringPrice`` that reads an undeclared row has it named in the reason
    (``; unpriced: ``), so a price that ties or orders the two on a missing row says so."""
    b, at = Lowering(form, B_STATIONARY), Lowering(form, AT_STATIONARY)
    pb, pat = float(price(b)), float(price(at))
    priced = f"B stationary {pb:.6g} J ({b.cells} bank cells), A^T stationary {pat:.6g} J ({at.cells} bank cells)"
    if isinstance(price, LoweringPrice) and (named := tuple(dict.fromkeys((*price.unpriced(b), *price.unpriced(at))))):
        priced += f"; unpriced: {', '.join(named)}"
    if not cost_only:
        return b, (f"B stationary, the declared default: the choice is function-bearing ({why}), so price does not "
                   f"switch it; priced {priced}")
    chosen = min((b, at), key=lambda low: (float(price(low)), low.cells, low.stationary != B_STATIONARY))
    return chosen, f"{chosen.stationary} stationary, the cheaper (cost-only choice): {priced}"


def bank_indices(form: OperandForm) -> Sequence[Tuple[int, ...]]:
    """Every index of the batch axes, in row-major order (one bank each)."""
    out: List[Tuple[int, ...]] = [()]
    for extent in form.batch:
        out = [(*prefix, i) for prefix in out for i in range(extent)]
    return out
