"""FX rewrites applied before conversion, one module each, dispatched by ``graph_normalization.normalize_fx_graph``
(and leaf-call inlining by the tracer, ``torch_graph_tracer.trace_model``, before writes and shapes are read).

``leaf_call_inlining``: an opaque call of a Python function or a bound method of one (an FX leaf function) whose
operands other than tensors are literals, bound by its code (``operand_samples.code_call``: never a signature it
reports), is replaced by its body as TorchDynamo captures it (``body_capture``: never run, on stand-in operands
whose metadata the seeded runs at batches 1, 2 and 3 give, ``operand_samples``; the comparisons at batch 1 and where a
symbol is 0 or 1 in ``body_comparison``, the bindings, normalization and return value in ``body_output``), in traced
order on a trial copy, the operands' extents proved in the converter's batch domain (``leaf_operands``); it stays
opaque, its reason recorded, when the capture is not one graph, decides on the batch, takes another path at such a
batch, writes in place or holds a node outside the declared set, when an operand is not a tensor computed by declared
operators, its extents are not proved, it is a distinct tensor of another operand's storage, or it is neither a traced
value nor a declared literal, when its target is another callable (a ``functools.partial``, a callable object) or the
signature its code gives is variadic or does not bind, or when it runs under a torch-function mode.
``chunk_to_narrow``: a ``chunk`` whose every user is an integer ``getitem`` becomes one ``narrow`` per user. The
batch-axis push-forward (``batch_axis_push_forward``): a permute moving the batch off axis 0 whose every consumer
indexes it back or permutes the batch back becomes a batch-leading form. ``sdpa_decomposition`` (under
``decompose_leaf_modules``): ``scaled_dot_product_attention`` becomes its declared operators. Each rewrite computes the
same values and the same views as the native graph, and touches no node a write in place orders (``rewritable``).
"""

from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import rewritable

__all__ = ["rewritable"]
