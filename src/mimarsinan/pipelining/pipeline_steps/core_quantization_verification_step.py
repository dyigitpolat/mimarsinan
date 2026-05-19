from __future__ import annotations

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.transformations.weight_quantization import TensorQuantization

import numpy as np


class CoreQuantizationVerificationStep(PipelineStep):
    """Fail fast if weight_quantization=True but IR NeuralCores are not chip-quantized."""

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
        if not self.pipeline.config.get("weight_quantization", False):
            cores = ir_graph.get_neural_cores()
            if cores:
                print(
                    "[CoreQuantizationVerificationStep] Skipping verification (weight_quantization=False); "
                    f"IR has {len(cores)} NeuralCores with float weights."
                )
            return

        bits = int(self.pipeline.config["weight_bits"])
        quantizer = TensorQuantization(bits)

        cores = ir_graph.get_neural_cores()
        if not cores:
            print("[CoreQuantizationVerificationStep] No NeuralCores found in IRGraph (nothing to verify).")
            return

        failures = []
        scale_tol = 1e-6
        bank_checked: set[int] = set()
        for core in cores:
            ps = core.parameter_scale
            try:
                scale = float(ps.item())
            except Exception:
                scale = float(ps)

            if scale == 0.0:
                failures.append(f"{core.name}: parameter_scale is 0")
                continue
            if abs(scale - 1.0) > scale_tol:
                failures.append(
                    f"{core.name}: parameter_scale={scale} (expected 1.0 after chip quantization)"
                )
                continue

            bank_id = getattr(core, "weight_bank_id", None)
            if bank_id is not None:
                if bank_id in bank_checked:
                    continue
                bank_checked.add(bank_id)

            mat = core.get_core_matrix(ir_graph)
            dtype = getattr(mat, "dtype", None)
            if dtype is not None and np.issubdtype(dtype, np.integer):
                maxv = int(mat.max()) if mat.size else 0
                minv = int(mat.min()) if mat.size else 0
                if maxv > quantizer.q_max:
                    failures.append(
                        f"{core.name}: max(W)={maxv} > q_max={quantizer.q_max}"
                    )
                if minv < quantizer.q_min:
                    failures.append(
                        f"{core.name}: min(W)={minv} < q_min={quantizer.q_min}"
                    )
                continue

            scaled = mat * scale
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


