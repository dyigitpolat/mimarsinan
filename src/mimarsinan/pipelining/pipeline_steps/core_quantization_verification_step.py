from __future__ import annotations

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.transformations.weight_quantization import TensorQuantization

import numpy as np


class CoreQuantizationVerificationStep(PipelineStep):
    """
    Verify that *mapped neural cores* are quantized.

    This is intentionally placed:
      Normalization Fusion -> Soft Core Mapping (IRGraph) -> THIS STEP -> CoreFlow Tuning

    Rationale:
    - CoreFlow tuning should not be responsible for verifying weight quantization.
    - We want a clear failure if a model is not quantized before tuning/compilation.
    """

    def __init__(self, pipeline):
        requires = ["ir_graph"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        ir_graph = self.get_entry("ir_graph")
        bits = int(self.pipeline.config["weight_bits"])
        quantizer = TensorQuantization(bits)

        cores = ir_graph.get_neural_cores()
        if not cores:
            print("[CoreQuantizationVerificationStep] No NeuralCores found in IRGraph (nothing to verify).")
            return

        failures = []
        for core in cores:
            ps = core.parameter_scale
            try:
                scale = float(ps.item())
            except Exception:
                scale = float(ps)

            if scale == 0.0:
                failures.append(f"{core.name}: parameter_scale is 0")
                continue

            scaled = core.core_matrix * scale
            rounded = np.round(scaled)

            maxv = float(np.max(rounded))
            minv = float(np.min(rounded))

            if maxv > quantizer.q_max + 1e-6:
                failures.append(f"{core.name}: max(|round(W*scale)|)={maxv} > q_max={quantizer.q_max}")
            if minv < quantizer.q_min - 1e-6:
                failures.append(f"{core.name}: min(round(W*scale))={minv} < q_min={quantizer.q_min}")

            if not np.allclose(scaled, rounded, atol=1e-3, rtol=1e-3):
                failures.append(f"{core.name}: W*scale is not integer (not allclose to round)")

        if failures:
            msg = "[CoreQuantizationVerificationStep] Quantization verification FAILED:\n" + "\n".join(
                f"  - {e}" for e in failures[:50]
            )
            if len(failures) > 50:
                msg += f"\n  ... (+{len(failures)-50} more)"
            raise AssertionError(msg)

        print(f"[CoreQuantizationVerificationStep] OK: verified {len(cores)} NeuralCores at {bits}-bit quantization.")


