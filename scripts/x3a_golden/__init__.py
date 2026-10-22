"""X3a model-level golden and independent random packer (plan section 10.2, second agent).

Written from the theorem box's semantics (the memos of plan section 10.1) and from nothing in the
toolchain: no module here imports ``mimarsinan`` (a fresh-interpreter test asserts it). Modules:

- ``program``: logical integer networks, declarations, segments and fused timing;
- ``kernels``: the integer kernels (staircase, fire and reset, saturation, encodes, laws);
- ``reference``: the identity single-pass semantics and the integer network N;
- ``packer``, ``schedule``: the random packer drawing every X3a cost-only decision (placement
  permutation, neuron split, exact fan-in fusion, coalescing, pass partition and order, residency,
  axon-slot assignment) and the identity packing;
- ``machine``: the packed program executed from its slot memory;
- ``hypotheses``: the hypotheses with their assertion sites, off switches and exercise tests;
- ``generator``, ``screen``: the seeded generator and the screen with Clopper-Pearson bounds;
- ``witnesses``: one hand-built witness per hypothesis (difference when off, equality when on).
"""

SCHEMA = "x3a-golden/1"
