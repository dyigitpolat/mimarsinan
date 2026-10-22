"""The symbolic-batch shape pass: every traced value in the affine batch domain, and the opaque batch values (review
round 14, the owner's shape-proof domain; ``propagation.py`` runs it, ``kernel_step.py`` is its step for a kernel).

The pass runs for a PURE program only (``purity.py``: declared operators, declared value types); an impure program has
no static shape proof and the pass never runs for it. It walks the traced graph once, in order, and gives every value
its domain value (``batch_domain``): the input's axis 0 is the batch B (``BATCH``), a parameter's extents are integers.

- A scalar operator (``CONVERSION_TIME_OPERATORS``) is computed by the domain (``batch_domain.evaluate``); its value is
  an integer, an affine form ``a*B + c`` or ``Opaque``.
- A shape query (``size``, ``shape``, ``dim``, ``ndim``, ``numel``) reads its tensor's domain extents
  (``batch_domain.shapes.query``); ``dtype`` reads the native dtype the tracer's shape propagation recorded.
- A tensor kernel's output extents are the ones its declared batch rule states in the domain (``batch_domain.kernels``,
  ``kernel_rules``, ``module_rules``, ``layouts``, ``selections``) on its call bound to the kernel's signature
  (``batch_domain.binding``, review round 16), accepted where torch's runs agree at two batches at least, the
  FakeTensor run at the hint batch (``SYMBOLIC_HINT``) and the seeded traces at batches 1, 2 and 3
  (``batch_domain.acceptance``; [corrected in review round 16: the hint alone]); an extent no rule states, or one
  torch's runs contradict, is ``Opaque``. A kernel with no batch-dependent operand is batch-free: torch computes it from
  integers, as natively.

Torch's kernels run on FakeTensors only as the vehicle of the program (``decision_values.py``): their operands are built
from the domain, their results are never read for a fold. Torch's symbolic results (sympy, ShapeEnv, guards) are used
only for the cross-check, which makes a disagreeing extent opaque.

Opaque values. A value the domain does not compute is a decision (``Decision``): kind ``opaque``, or ``failed`` when a
kernel's run on a batch-dependent operand raised. Every scalar or extent computed from it is opaque with the same
decision (``Opaque.origin``), so the pass computes which per-sample shapes it reaches (``shape_decisions.py``): such a
value is computed on the host at run time, a layout method it reaches is the host op of its native method, a static
per-sample mapper that reads such a layout is refused by name (``LayoutMixin._require_static_reads``), and the
program's stage 2 is refused by name, naming the first opaque node (``OpaqueBatchValueError``). A data-dependent size
(an unbacked symbol of torch's run, ``nonzero``) is no decision: it is never folded, and a layout it reaches is a
violation (``batch_layout.py``).

Soundness lemma (pass). For a pure program under the trust base (T1, T2, T3) and trace fidelity: every integer extent and
scalar the pass holds is the native one at every batch B >= 1 at which the program runs, and every affine one is
``a*B + c`` there. By induction over the traced order of the graph the tracer rewrote (``operand_writes``, review round
15: no node reads an operand after a write of it, so a node's value is the native value its node computes): inputs
and parameters by construction; a scalar operator by the domain's lemma (``batch_domain/operations.py``); a kernel by
the kernels' lemma (``batch_domain/kernels.py``) and the acceptance lemma (``batch_domain/acceptance.py``); an opaque
value states nothing. The pass runs in fp64 whatever the model's dtype: shapes do not depend on the dtype, and the
dtype a program reads is the native one.
"""

from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import (
    SYMBOLIC_HINT,
    Decision,
    Extent,
    SymbolicPass,
    SymbolicShape,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.symbolic_batch.propagation import symbolic_shapes
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import (
    graph_device,
    placeholder_metas,
    rng_neutral,
)

__all__ = [
    "Decision", "Extent", "SYMBOLIC_HINT", "SymbolicPass", "SymbolicShape", "graph_device", "placeholder_metas",
    "rng_neutral", "symbolic_shapes",
]
