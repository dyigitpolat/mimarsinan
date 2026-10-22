"""FX rewrites applied before conversion, one module each, dispatched by ``graph_normalization.normalize_fx_graph``.

``chunk_to_narrow``: a ``chunk`` whose every user is an integer ``getitem`` becomes one ``narrow`` per user. The
batch-axis push-forward (``batch_axis_push_forward``): a permute moving the batch off axis 0 whose every consumer
indexes it back or permutes the batch back becomes a batch-leading form. ``sdpa_decomposition`` (under
``decompose_leaf_modules``): ``scaled_dot_product_attention`` becomes its declared operators. Each rewrite computes the
same values and the same views as the native graph, and touches no node a write in place orders (``rewritable``).
"""

from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import rewritable

__all__ = ["rewritable"]
