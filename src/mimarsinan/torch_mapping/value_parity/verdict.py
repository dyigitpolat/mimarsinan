"""A probe stage's verdict: the measured comparison, or the failure that left none, and its named refusal."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

from mimarsinan.torch_mapping.conversion_probe import ConversionProbeError
from mimarsinan.torch_mapping.value_parity.rounding import REFERENCE_FLOOR_MARGIN

STAGE_NATIVE = "native reference"
STAGE_CONVERTED = "converted model"
STAGE_VALUE_PROGRAM = "identity value program"


@dataclass(frozen=True)
class ValueParityResult:
    """One stage's verdict; ``reason`` carries a failure that left no measurement."""

    ok: bool
    context: str
    stage: str
    contract: str
    bound: float
    samples: int
    weight_layer_depth: Optional[int] = None
    randomized_parameters: Tuple[str, ...] = ()
    reference_constant: bool = False
    escalated: bool = False
    compared_on: Optional[str] = None
    reassociation_bound: Optional[float] = None
    max_abs_error: Optional[float] = None
    reference_max: Optional[float] = None
    relative_error: Optional[float] = None
    first_diverging_node: Optional[str] = None
    first_diverging_operator: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[BaseException] = None
    wall_s: float = 0.0
    reference_floor: Optional[float] = None  # the native program's own floor on the refused inputs (rounding.py)

    @property
    def reference_undetermined(self) -> bool:
        """A refusal that is the reference's: the native program is not reproducible within the bound on the compared
        inputs (``reference_floor`` above it) and the error lies within ``REFERENCE_FLOOR_MARGIN`` times that floor,
        so the comparison cannot decide at the bound; no conversion defect is shown, none is excluded below it. The
        floor and the error must both be finite: an infinite or undefined one bounds nothing, so the refusal is a
        value mismatch [review round 1 of the coverage fixes: an infinite floor held every error undecidable]."""
        floor, error = self.reference_floor, self.relative_error
        return (not self.ok and floor is not None and error is not None and math.isfinite(floor)
                and math.isfinite(error) and floor > self.bound and error <= REFERENCE_FLOOR_MARGIN * floor)

    def _undetermined(self) -> str:
        return (f"; the native model itself is not reproducible within the bound on these inputs: with every weight "
                f"layer's reduction reversed it differs from itself by {self.reference_floor:.3e}, so the comparison "
                f"cannot decide at the bound") if self.reference_undetermined else ""

    def _node(self) -> str:
        if not self.first_diverging_node:
            return "first diverging node not located"
        operator = f" ({self.first_diverging_operator})" if self.first_diverging_operator else ""
        return f"first diverging node {self.first_diverging_node!r}{operator}"

    def format(self) -> str:
        head = f"[{self.context}] value parity {'PASS' if self.ok else 'REFUSED'} ({self.stage} vs {self.contract})"
        if self.reason is not None:
            node = f"; {self._node()}" if self.first_diverging_node else ""
            return f"{head}: {self.reason}{node}"
        relation = "<=" if self.ok else ">"
        randomized = (
            f", randomized {len(self.randomized_parameters)} all-zero parameters"
            if self.randomized_parameters else ""
        ) + (", reference constant across the inputs" if self.reference_constant else "")
        inputs = f"{self.samples} seeded fp64 inputs and the first at batch 1" + (
            f", {self.samples} escalated-magnitude inputs, {self.samples} intermediate-magnitude inputs"
            if self.escalated else "")
        where = f" on {self.compared_on}" if self.compared_on and not self.ok else ""
        beta = "" if self.reassociation_bound is None else f", reassociation {self.reassociation_bound:.3e}"
        floor = "" if self.reference_floor is None else f", reference floor {self.reference_floor:.3e}"
        measured = (
            f"relative error {self.relative_error:.3e} {relation} bound {self.bound:.3e}{where} "
            f"(D(P)={self.weight_layer_depth}{beta}{floor}; max|err|={self.max_abs_error:.3e}, "
            f"max|ref|={self.reference_max:.3e}, {inputs}{randomized}, {self.wall_s:.2f} s)"
        )
        return f"{head}: {measured}" if self.ok else f"{head}: {measured}{self._undetermined()}; {self._node()}"


def describe_failure(exc: BaseException) -> str:
    """``Type: first line`` of a failure, with its root cause when it was re-raised."""
    root = exc
    while root.__cause__ is not None:
        root = root.__cause__
    text = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    if root is not exc:
        text += f" (root cause {type(root).__name__}: {str(root).splitlines()[0] if str(root) else ''})"
    return text


def degenerate_reference(base: ValueParityResult, seeded: str, escalated: str) -> ValueParityResult:
    """The refusal of a reference uninformative on the seeded inputs (``seeded``: zero, non-finite or constant)
    with no finite non-zero entry on the escalated ones (``escalated``)."""
    return replace(base, stage=STAGE_NATIVE, reference_max=0.0, reason=(
        f"the native reference output is {seeded} on all {base.samples} seeded inputs and {escalated} on all "
        f"{base.samples} escalated-magnitude inputs (after randomizing {len(base.randomized_parameters)} all-zero "
        "parameters): neither set has an informative entry, so comparing it carries no information; the "
        "conversion cannot be certified"
    ))


class ConversionValueParityError(ConversionProbeError):
    """A conversion whose output differs from its contracted function (or cannot be compared), named."""

    def __init__(self, result: ValueParityResult):
        self.result = result
        self.stage = result.stage
        self.max_abs_error = result.max_abs_error
        self.reference_max = result.reference_max
        self.relative_error = result.relative_error
        self.first_diverging_node = result.first_diverging_node
        self.first_diverging_operator = result.first_diverging_operator
        super().__init__(result.format())

    @staticmethod
    def of(result: ValueParityResult) -> "ConversionValueParityError":
        """The refusal of a refused result: ``ReferenceUndeterminedError`` where the reference's floor decides it."""
        return (ReferenceUndeterminedError if result.reference_undetermined else ConversionValueParityError)(result)


class ReferenceUndeterminedError(ConversionValueParityError):
    """A comparison the probe cannot decide at its bound: the native model is not reproducible within the bound on the
    compared inputs, and the error lies within a small multiple of the native model's own floor there
    (``ValueParityResult.reference_undetermined``). Named apart from a value mismatch, which it is not shown to be."""


ShapeLiterals = Mapping[str, Tuple[str, str]]
"""A conversion's folded shape values: FX node name -> (operator, ``repr`` of the literal)."""


def literal_differences(returned: ShapeLiterals, witness: ShapeLiterals) -> Tuple[Tuple[str, str, str, str], ...]:
    """``(node, operator, returned, witness)`` for every FX node whose folded literal differs between the two."""
    differences = []
    for name in sorted(set(returned) | set(witness)):
        got, seen = returned.get(name), witness.get(name)
        if got != seen:
            operator = (got or seen or ("?", ""))[0]
            differences.append((name, operator, got[1] if got else "no literal", seen[1] if seen else "no literal"))
    return tuple(differences)


class WitnessProgramMismatchError(ConversionProbeError):
    """The probe's witness conversion is of another program than the returned conversion (review round 13,
    ``program_binding.py``: ``of_program``), or folds other conversion-time literals, named by FX node."""

    def __init__(self, differences: Tuple[Tuple[str, str, str, str], ...], *, program: Optional[str] = None):
        self.differences = differences
        listed = "; ".join(
            f"{operator} (FX node {name!r}): {got} in the returned conversion, {seen} in the witness"
            for name, operator, got, seen in differences
        )
        what = (f"is of another program than the returned conversion ({program})" if program is not None else
                f"folds other conversion-time literals than the returned conversion ({listed})")
        super().__init__(
            f"the probe's witness conversion {what}: its certificate would cover a different program, so the "
            "conversion is refused"
        )

    @classmethod
    def of_program(cls, node: str, operator: str, difference: str) -> "WitnessProgramMismatchError":
        """The first FX node at which the returned and the witness traced programs differ, and how."""
        return cls((), program=f"{operator} (FX node {node!r}): {difference}")


class WitnessProgramError(ConversionProbeError):
    """The probe's witness program (the traced graph with its parameters and buffers in fp64, review round 14) does
    not run in fp64: a program that holds its parameters' dtype as a literal (``x.to(next(self.parameters()).dtype)``)
    computes in another dtype than its fp64 parameters."""

    def __init__(self, failure: str):
        self.failure = failure
        super().__init__(
            f"the probe's witness program, the traced graph with its parameters and buffers in fp64, does not run in "
            f"fp64 ({failure}), so the conversion cannot be certified"
        )
