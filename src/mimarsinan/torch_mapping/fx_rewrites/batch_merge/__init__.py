"""The batch-merge push-forward (coverage W7): a merged value, ``(B * n, ...)``, carried split, batch-leading
(``push_forward``), by the axis rules of the operations that read it (``axes``, ``layouts``)."""

from mimarsinan.torch_mapping.fx_rewrites.batch_merge.push_forward import push_batch_merge_forward

__all__ = ["push_batch_merge_forward"]
