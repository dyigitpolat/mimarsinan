"""The rules a captured leaf-call body is judged and rewritten by before it is spliced (coverage W8, sign-off L8):
``guard_proof`` proves the capture's statements on the batch in the affine batch domain, ``symbolic_record`` records
torch's symbolic operations and static evaluations during a capture and re-proves each there, ``folds`` the values
Dynamo folds into the body's Python from a symbolic-shape query or a tracing-state function, ``local_writes``
computes a body's assignments to tensors it allocates out of place."""

from mimarsinan.torch_mapping.fx_rewrites.body_rules.guard_proof import prove
from mimarsinan.torch_mapping.fx_rewrites.body_rules.local_writes import rewrite_assignments
from mimarsinan.torch_mapping.fx_rewrites.body_rules.symbolic_record import recorded, unproved_record

__all__ = ["prove", "recorded", "rewrite_assignments", "unproved_record"]
