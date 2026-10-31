"""Registry entries: the declared host-kernel evaluation of the deployed program (U-PS)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.mapping.support.host_kernel_evaluation import HOST_KERNEL_EVALUATIONS

ENTRIES = (
    _E("host_kernel_evaluation", group="mapping_strategy", owner="mapping/support/host_kernel_evaluation",
       type=T.ENUM, options=HOST_KERNEL_EVALUATIONS, category=Category.ADVANCED, exposure="user",
       label="Host Kernel Evaluation",
       effect="Evaluates every host kernel of the deployed program on its batch or one sample at a time",
       doc="per_sample: every host kernel of the deployed program (its host convolutions, matmuls, norms and the "
           "rest) is evaluated one sample at a time, the results concatenated in order, by every executor and by "
           "the NF twin wherever a parity gate compares it with the program, so a sample's host values are a "
           "function of that sample alone: Theorem 1's host condition by construction (an fp32 host kernel on its "
           "batch is batch-dependent at the ulp level). A kernel that cannot run on one sample alone is refused by "
           "name. batched: every host kernel on its batch.",
       empty_means="batched — every host kernel on its batch, every program byte for byte"),
)
