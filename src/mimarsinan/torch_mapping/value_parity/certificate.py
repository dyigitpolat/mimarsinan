"""Conversion certificate: the probe's verdict and the converter version, stamped on a converted flow at birth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

CONVERTER_VERSION = "wave2-cov-w78-int1"
"""Bumped whenever a conversion rule changes the converted function: a flow stamped otherwise is stale."""

CERTIFICATE_HYPOTHESES: Tuple[str, ...] = (
    "trace fidelity: the native model's output must not depend on Python-level identity, hashing, type checks "
    "(isinstance, type, callable, hasattr) or string forms of tensor shapes",
    "T1 shipped runtime: Python and torch run as shipped (nothing of torch is rebound or rewritten after mimarsinan "
    "is imported, and no user mode or subclass poses as torch's own)",
    "T2 nn.Module call protocol: the model is called through nn.Module's own call protocol (no class in its hierarchy "
    "redefines attribute lookup or nn.Module's private call machinery, and no user container replaces nn.Module's "
    "private tables or torch's global hook tables)",
    "T3 opaque-helper effects: an opaque user callable (a torch.fx.wrap function or a leaf module whose body the "
    "trace does not enter) has the effects on its tensor arguments, writes and storage aliasing, that the probe "
    "observes, made through torch's version-counted operations",
)
"""The stated boundary of every certificate, not a tolerance: trace fidelity and the trust base T1, T2 and T3.

T3, opaque-helper effects (the owner's decision after review round 15): an opaque user callable is a ``torch.fx.wrap``
function or a leaf module whose body the trace does not enter (a ``torch.fx.wrap`` function that leaf-call inlining
splices in, ``fx_rewrites/leaf_call_inlining.py``, is none: its body is TorchDynamo's capture of it,
``fx_rewrites/body_capture.py``, which interprets its bytecode on stand-in operands and never runs it, spliced only
when the capture is one graph, no guard of it on the batch fails at some batch B >= 1, the capture at batch 1 and at
every batch where a batch symbol is 0 or 1 is the same program, the body writes nothing in place (a write torch
declares or an in-place operator's name names), every node is a declared operator and every operand is a tensor
computed by declared operators, its extents proved affine in the converter's batch domain and equal to the seeded runs
at batches 1, 2 and 3, the operands that are one tensor natively one stand-in, no two distinct ones of one storage
[restated in the targeted pass of fix round 2: the forms were the seeded runs' fit alone, and the batch-1 capture the
only comparison]; the spliced body is root code for every rule after it, and Dynamo evaluates a type check of a tensor as for a real one,
refuses a string form or a hash of one, and holds trace-time state, a global read included, as FX holds the root's;
[restated in fix round 2 of the coverage fixes: round 1 traced the body once on FX's Proxies]); its effects on its
tensor arguments, writes and storage aliasing, are those the probe observes (the tracer's run at batch 1, ``symbolic/operand_writes/observed.py``), made through
torch's version-counted operations. Effects the probe sees are converted exactly, in the native order of reads and
writes, or refused by name (``OperandWriteError``; ``HostOpWriteError`` in the converted forward). Effects it does not
see are witnesses of the boundary, never claimed covered: a write that bypasses the version counter (``Tensor.data``, a
numpy view), and an alias or a write that appears only at batch sizes the probe did not run.

T1 and T2, the trust base (the owner's decision after review round 10). T1, shipped runtime: Python and torch run as
shipped: no function, class, code object, closure cell, default or module global of torch is rebound or rewritten after
mimarsinan is imported, and no user mode or subclass poses as torch's own. T2, nn.Module call protocol: the model is
called through nn.Module's own call protocol: no class in the model's hierarchy redefines attribute lookup
(``__getattribute__``, a metaclass) or nn.Module's private call machinery (``__call__``, ``_call_impl``,
``_wrapped_call_impl``, ``_compiled_call_impl``), and no user container replaces nn.Module's private tables
(parameters, buffers, modules, hook tables) or torch's global hook tables. A program that violates T1 or T2 is outside
the guarantee: the forms the toolchain detects (the import snapshot of torch's code, ``symbolic/imported_code.py``; the
root call read as nn.Module's call reads it, ``symbolic/module_call.py``) are witnesses of the boundary, never claimed
covered. Everything a model does through public Python and torch interfaces is inside it: hooks and global hooks
registered through the public API, instance forwards, tensor, ``int`` and ``float`` subclasses, user helpers and user
modules, custom autograd functions, ``__torch_function__`` overrides and modes registered through the public API, and
data-dependent shapes.

Trace fidelity: FX tracing evaluates some Python operations on a Proxy before the converter runs (``x.size(0) is 2``,
``x.size(0) in {7, 13}``, ``isinstance(x.size(0), int)``, ``callable(x.size(0))``, ``repr(x.shape)``) and records
their trace-time result as a constant; a program whose output depends on one converts to a program that differs from
it at batch sizes the probe does not run, on w0-base as here. In an inlined helper's body Dynamo answers a type check as
for a real tensor, guards a membership test of an extent (so the call stays opaque) and refuses a string form or a
hash, so of these only identity (``n is 9``, ``id``), which it evaluates at trace time, stays in this class. A literal operand FX would re-type (a user subclass of
``tuple``, ``list`` or ``dict``, recorded as the plain container) is refused by the tracer (``TracingError``), so it is
not in this class, and so is a model whose native call is not the traced program (``RootCallError``: a hook on the root
or a global module hook, an instance-bound ``forward`` or a class's own ``__call__``, which FX never runs, or a forward
parameter after the first that the program reads). User code and user types that FX records are not in it either: a
program with a call outside the declared operator set, with a literal operand, a ``get_attr`` value, a parameter, a
buffer, a module attribute or a module table outside the declared types (an ``int`` subclass with its own
``__rmul__``, a tensor subclass with its own ``__torch_function__``, a ``dict`` subclass as ``_parameters``), or
converted while torch's declared code is not the one imported (a rebound global of torch's own module, a user mode),
is impure (``symbolic/purity.py``). The symbolic-batch pass never runs for it, nothing of it is folded or computed with
at conversion, every shape value and every layout operation of it is the host op of its native function on its
run-time operands, a module outside the set is the host op of its own call (every module, while torch's code is not
the one imported), and its value program, which fixes every per-sample shape, is refused by name at stage 2
(``ImpureShapeProgramError``)."""


class UnverifiedConversionWarning(UserWarning):
    """A permissive (``strict=False``) conversion returned a flow its probe did not certify."""


@dataclass(frozen=True)
class StageVerdict:
    """One probe stage's measured outcome (deterministic: no wall time)."""

    stage: str
    ok: bool
    bound: float
    samples: int
    relative_error: Optional[float] = None
    max_abs_error: Optional[float] = None
    reference_max: Optional[float] = None
    first_diverging_node: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_result(cls, result) -> "StageVerdict":
        return cls(
            result.stage, bool(result.ok), float(result.bound), int(result.samples),
            result.relative_error, result.max_abs_error, result.reference_max,
            result.first_diverging_node, result.reason,
        )


@dataclass(frozen=True)
class ConversionCertificate:
    """``certified``: every stage that ran passed; ``missing_stages``: required stages that did not run;
    ``shape_proof``: what the converted program's static shapes are proved by (``static shape proof: every
    batch-dependent size in the affine batch domain ...``, with the first opaque batch value computed on the host at
    run time when there is one, review round 14; ``no static shape proof: impure program at ... (...)`` for an impure
    program; ``None`` when a pure program's conversion asked for no shape, or for a certificate stamped before the
    field)."""

    converter_version: str
    contract: Optional[str]
    weight_layer_depth: Optional[int]
    required_stages: Tuple[str, ...]
    stages: Tuple[StageVerdict, ...]
    reason: Optional[str] = None
    shape_proof: Optional[str] = None

    @property
    def certified(self) -> bool:
        return self.reason is None and bool(self.stages) and all(v.ok for v in self.stages)

    @property
    def hypotheses(self) -> Tuple[str, ...]:
        """The hypotheses the certificate holds under (``CERTIFICATE_HYPOTHESES``; not a field, so pickles keep)."""
        return CERTIFICATE_HYPOTHESES

    @property
    def missing_stages(self) -> Tuple[str, ...]:
        ran = {verdict.stage for verdict in self.stages}
        return tuple(stage for stage in self.required_stages if stage not in ran)

    @classmethod
    def from_results(
        cls, results: Sequence, required_stages: Sequence[str], shape_proof: Optional[str] = None,
    ) -> "ConversionCertificate":
        first = results[0] if results else None
        return cls(
            CONVERTER_VERSION, getattr(first, "contract", None), getattr(first, "weight_layer_depth", None),
            tuple(required_stages), tuple(StageVerdict.from_result(r) for r in results), shape_proof=shape_proof,
        )

    @classmethod
    def refused(
        cls, reason: str, required_stages: Sequence[str], shape_proof: Optional[str] = None,
    ) -> "ConversionCertificate":
        """The stamp of a permissive conversion whose structural probe failed before any value stage ran."""
        return cls(CONVERTER_VERSION, None, None, tuple(required_stages), (), reason, shape_proof)

    def summary(self) -> str:
        """The one-line verdict; a certified one names the hypotheses it holds under (``statement`` states them)."""
        stages = ", ".join(
            f"{v.stage} {'PASS' if v.ok else 'REFUSED'}"
            + (f" {v.relative_error:.3e} <= {v.bound:.3e}" if v.ok and v.relative_error is not None else "")
            for v in self.stages
        )
        names = [hypothesis.split(":")[0] for hypothesis in self.hypotheses]
        verdict = f"certified under {', '.join(names[:-1])} and {names[-1]}" if self.certified else "NOT certified"
        why = f"; {self.reason}" if self.reason else ""
        missing = f"; not run: {', '.join(self.missing_stages)}" if self.missing_stages else ""
        return (
            f"conversion {verdict} (converter {self.converter_version}; "
            f"{stages or 'no value stage ran'}{missing}{why})"
        )

    def statement(self) -> str:
        """The certification statement: the verdict (``summary``), what the static shapes are proved by, and every
        hypothesis in full (``CERTIFICATE_HYPOTHESES``); an uncertified conversion claims none."""
        claims = ["under the hypotheses:", *(f"- {hypothesis}" for hypothesis in self.hypotheses)] if self.certified \
            else ["no hypothesis is claimed: the conversion is not certified"]
        return "\n".join([self.summary(), f"shape proof: {self.shape_proof or 'no shape was asked for'}", *claims])


def certificate_problem(certificate: Optional[ConversionCertificate]) -> Optional[str]:
    """Why a converted flow may not be used as certified; ``None`` for a current, certified stamp."""
    if certificate is None:
        return (
            "it carries no conversion certificate: it was converted before the value-parity probe existed, or "
            "outside convert_torch_model"
        )
    if certificate.converter_version != CONVERTER_VERSION:
        return (
            f"its certificate is stale: converter {certificate.converter_version!r}, current "
            f"{CONVERTER_VERSION!r} (a conversion rule changed since)"
        )
    if not certificate.certified:
        return f"its {certificate.summary()}"
    if certificate.missing_stages:
        return (
            f"its conversion skipped required probe stages ({', '.join(certificate.missing_stages)}): "
            f"{certificate.summary()}"
        )
    return None
