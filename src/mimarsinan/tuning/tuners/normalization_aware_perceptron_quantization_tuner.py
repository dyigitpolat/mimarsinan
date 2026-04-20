"""Weight quantization tuner using NormalizationAwarePerceptronQuantization.

LSQ integration (Phase C1)
--------------------------
Before the trainer is built, we eagerly install an :class:`LSQQuantizer`
on every perceptron in the model.  That way ``PerceptronTransformTrainer``'s
``copy.deepcopy(self.model)`` picks up the quantizer on the aux (latent)
model as well, and the per-step transform (which reseeds and reuses the
quantizer rather than creating a new one) can accumulate learnt
``log_scale`` updates across steps rather than reverting to the
closed-form seed on every call.
"""

import torch

from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.lsq_quantization import LSQQuantizer
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner


class NormalizationAwarePerceptronQuantizationTuner(PerceptronTransformTuner):
    def __init__(self, pipeline, model, quantization_bits, target_accuracy, lr, adaptation_manager):
        self._install_lsq_quantizers_on(model, int(quantization_bits), pipeline.config["device"])
        super().__init__(pipeline, model, target_accuracy, lr)
        self.quantization_bits = int(quantization_bits)
        self.adaptation_manager = adaptation_manager

    @staticmethod
    def _install_lsq_quantizers_on(model, bits, device):
        """Attach an LSQQuantizer to every perceptron in ``model`` and
        seed its step from the current effective weight statistics.

        Doing this *before* :class:`PerceptronTransformTrainer` deepcopies
        the model is crucial: the aux (latent) model must carry identical
        LSQ quantizers so the optimiser's updates to ``log_scale`` survive
        the per-step ``_update_and_transform_model`` refresh.
        """
        for perceptron in model.get_perceptrons():
            q = getattr(perceptron, "weight_quantizer", None)
            if isinstance(q, LSQQuantizer) and q.bits == bits:
                continue
            w = PerceptronTransformer().get_effective_weight(perceptron)
            b = PerceptronTransformer().get_effective_bias(perceptron)
            p_max = max(
                float(torch.max(torch.abs(w)).item()),
                float(torch.max(torch.abs(b)).item()),
                1e-12,
            )
            new_q = LSQQuantizer(bits=bits).to(device)
            new_q.init_from_tensor(torch.tensor([p_max]))
            perceptron.set_weight_quantizer(new_q)

    def _get_previous_perceptron_transform(self, rate):
        return lambda perceptron: None

    def _get_new_perceptron_transform(self, rate):
        def transform(perceptron):
            NormalizationAwarePerceptronQuantization(
                self.quantization_bits,
                self.pipeline.config["device"],
                rate,
            ).transform(perceptron)
        return transform

    def _update_and_evaluate(self, rate):
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        with torch.no_grad():
            self.trainer._update_and_transform_model()
        # LSQ weight transforms are coarser than per-activation mixing, so
        # per-cycle decisions need the full-tier probe rather than the
        # cheaper fast-tier one used elsewhere.
        return self.trainer.validate_full()

    def run(self):
        return super().run()
