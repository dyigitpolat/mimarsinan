"""Operand writes: a call that writes a tensor operand in place is a new definition of that operand, run in the native
order of reads and writes (review rounds 15 and 16).

FX records a write in place (``torch.resize_as_(z, y)``, ``torch.clamp_(y, a, b)``, ``torch.ops.aten.set_``,
``F.relu(y, inplace=True)``, ``nn.ReLU(inplace=True)``) as a call on the operand's node, and every later reader of the
operand reads that same node. Each node's shape was propagated at its own node, and the converted model runs its host
ops in an order of its own (the mapper graph's): a later reader would read the operand's shape before the write, and
an earlier one could run after it. The tracer applies, on the traced graph (``trace_model``):

1. **Definition** (``define_writes``). For every declared write that returns its operand (``declared.py``: torch's
   schema annotation ``Tensor(a!)``, ``inplace=True``, a module whose ``inplace`` flag is set), every node after the
   write that reads the operand reads the write's result in its place, natively the same tensor. So no shape or value
   of the operand before the write is read after it; dead-code elimination keeps a write whose operand a later node
   reads (``observable_write``).
2. **Storage** (``settle_writes``, on the tracer's shape propagation at batch 1, ``observed.py``). The values that may
   share the written operand's storage other than through the write's result, S (torch's alias annotations, and the
   storage the run shows shared): a node that reads one after a declared write is refused by name
   (``OperandWriteError``: it would read the value before the write); every node that reads one before the write is an
   ordering edge of the write (``RUN_AFTER``): the write's host op runs after it in the converted model
   (``Mapper.run_after``, honoured by the mapper graph's execution order), and the write acts on its operand itself,
   as natively; no Linear fusion moves such a read (``graph_normalization``). [changed in review round 16, targeted
   pass: the write acted on ``torch.clone(operand)``, which keeps the values and not the layout or the storage, and
   every later reader read the clone.] A call torch's declaration lets share its operand's storage (``Tensor(a)``:
   ``reshape``, ``contiguous``, ``flatten``, ``to``; ``declared.view_aliases``) whose result the run shows in another
   storage is a copy, no edge of S (review round 16: a ``reshape`` of a permuted view, which copies at every batch, was
   taken as sharing); where a rule relies on such a copy, the call is the host op of its native method, which checks at
   run time that its operand has the run's strides and that its result is again a copy (``COPY_GUARD``,
   ``ComputeOpMapper.copy_guard``, ``compute_modules.HostOpAliasError``).
3. **Observed** (``settle_writes``, ``check_record``). A call outside the declared operator set runs after every node
   that reads a value of its operands' storage before it (``RUN_AFTER``), on its operands themselves. Every node after
   such a call whose write of an operand the run shows, and that reads that operand, reads ``read_after(operand,
   call)`` instead (``compute_modules.read_after``: the operand itself, a value ordered after the call by data), as
   for a declared write that returns its operand; a node that reads another value of S defined before the write, after
   it, is refused by name, and so is an undeclared write of a declared call that another node reads after it. [changed
   in review round 16, targeted pass: such a write was refused, and an operand every other reader read before the call
   was given as a copy, ``CLONED_OPERANDS``.] After the rewrite a read of a value whose state (version counter, shape,
   strides, storage) changed since its node ran is refused by name, but ``read_after``'s read of the operand the call
   it follows wrote; and the host op of every call outside the declared operator set checks at run time that it
   writes no operand another node reads after it, other than one routed through ``read_after`` (``KEPT_OPERANDS``,
   ``ComputeOpMapper.kept_inputs``, ``compute_modules.unwritten_call``: ``HostOpWriteError``, a refusal by name in the
   converted forward). An inference tensor (``torch.inference_mode()``) keeps no version counter: its bytes stand in
   for it, in the run and in the guard (``compute_modules.write_counter``, review round 16).

Soundness lemma (operand writes, restated in review round 16, targeted pass). For a program under the trust base (T1,
T2, T3) and trace fidelity that the tracer does not refuse, the converted model reads, at every node, the value the
native program reads at that node, at every batch and input: every node reads its operands' values in an order of
the converted model in which, on every storage, each read comes after the writes it comes after natively and before
the writes it comes before natively. A read after a write reads the write's result, a value defined after the write
by data (the declared write's own result, or ``read_after(operand, call)``), so it runs after the write; no other value
of S is read after a write. A read before a write is an ordering edge of the write, so it runs before the write; a
write acts on its operand itself, so a later read sees the native storage, layout and contents. Every value is the
tensor its node's host op returns natively: no copy is inserted, and a call the conversion takes as a copy is checked
at run time to be one (``COPY_GUARD``). A write the run at batch 1 does not observe, of an operand another node reads
after the call, is refused by name in the converted forward (``HostOpWriteError``). A copy is a copy at every batch at
the run's strides: whether ``reshape``, ``contiguous`` or ``flatten`` returns a view is a function of the operand's
extents and strides, an axis of extent 1 sets no condition, and at a batch above 1 axis 0 adds one, so a call that
copies at batch 1 copies at every batch at the same strides, which the guard checks on the operand the converted
model computes with torch's own methods (``HostOpAliasError`` otherwise). Hypotheses:

- T3 (``value_parity/certificate.py``): the effects of a call outside the declared set on its tensor arguments are the
  ones the run at batch 1 observes, made through version-counted operations; a write through ``Tensor.data`` or a
  numpy view, and an alias or a write that appears only at another batch, are boundary witnesses;
- torch's declared operators share an operand's storage where their declaration says they may (``declared.py``), or
  alike at every shape (seen in the run at batch 1: ``F.dropout`` in eval, a permuting ``torch.einsum``, ``type_as``
  of the same dtype); ``Tensor.sum_to_size``, which returns its operand at some shapes with no annotation, is declared
  [corrected in review round 16: the hypothesis said "only where their schema annotates it, or at every shape alike",
  which ``sum_to_size`` falsifies; it has no conversion rule, so no converted program held one];
- the converted model's values share storage where the native ones do: a host op runs the native call on its run-time
  operands, and a mapper that lowers a layout method computes it with torch's ``reshape`` or ``permute``, a view
  wherever the native method returns one (a copy the conversion relies on is guarded, ``COPY_GUARD``);
- the copy guard's premise: the native operand of a guarded copy has, at every batch, its strides of the run at
  batch 1 with axis 0 of extent B, as torch lays out a value whose batch is its leading axis; the guard checks the
  operand the converted model computes against them.

[corrected in review round 16, targeted pass: the round-16 lemma said "the rewritten graph computes at every node the
native value", and a copy (``torch.clone``, ``Tensor.clone``) kept the values only: a helper that reads its operand's
layout (``is_contiguous``, ``stride``, ``storage_offset``) and a ``torch.as_strided`` of the written view's storage
computed another value at the helper's node, from batch 7 (LayoutClonedLate, PureAsStridedAfterWriteCopyLate)]
"""

from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.declared import (
    Write,
    declared_aliases,
    declared_writes,
    in_place_form,
    observable_write,
    schema_declares_no_alias,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.observed import WriteRecord, WriteRecorder
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.rewrite import (
    COPY_GUARD,
    KEPT_OPERANDS,
    RUN_AFTER,
    OperandWriteError,
    check_record,
    define_writes,
    settle_writes,
)

__all__ = [
    "COPY_GUARD", "KEPT_OPERANDS", "RUN_AFTER", "OperandWriteError", "Write", "WriteRecord", "WriteRecorder",
    "check_record", "declared_aliases", "declared_writes", "define_writes", "in_place_form", "observable_write",
    "schema_declares_no_alias", "settle_writes",
]
