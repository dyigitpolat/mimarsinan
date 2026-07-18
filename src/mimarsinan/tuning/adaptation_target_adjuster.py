import math

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory


def target_decay_from_validation_samples(n_samples: int) -> float:
    """Decay factor from validation set size: ``1 - max(1/√n, 0.001)``, clamped to [0.95, 0.999]."""
    d = 1.0 - max(1.0 / math.sqrt(max(1, int(n_samples))), 0.001)
    return max(0.95, min(0.999, d))


class AdaptationTargetAdjuster:
    def __init__(self, original_target, decay=0.999, floor_ratio=0.90, frozen=False):
        assert decay < 1.0
        assert decay > 0.5
        assert 0.0 < floor_ratio <= 1.0

        self.decay = decay
        self.growth = 1.0 / decay
        self.target_metric = original_target
        self.original_metric = original_target
        self.floor = original_target * floor_ratio
        # Frozen = the origin-anchored compact (calculus §13.2 L-B): the
        # target never relaxes on a miss — misses report honestly instead of
        # licensing drift.
        self.frozen = bool(frozen)

    @classmethod
    def from_pipeline(cls, original_target, pipeline):
        """Build adjuster with validation-set-sized decay; floor ratio is ``1 - degradation_tolerance``."""
        dp = DataLoaderFactory.for_pipeline(pipeline).create_data_provider()
        n = dp.get_validation_set_size()
        decay = target_decay_from_validation_samples(n)
        dt = float(pipeline.config.get("degradation_tolerance", 0.05))
        floor_ratio = 1.0 - dt
        # Raw key read: the predicate SSOT (retention_envelope.
        # origin_anchored_compact_active) sits below tuner_base in the import
        # graph; importing it here would cycle through orchestration.
        frozen = bool(pipeline.config.get("origin_anchored_compact", False))
        return cls(original_target, decay, floor_ratio, frozen=frozen)

    def update_target(self, new_metric):
        if self.frozen:
            return
        if new_metric >= self.target_metric:
            self.target_metric = min(
                self.target_metric * self.growth, self.original_metric
            )
        else:
            self.target_metric = self.target_metric * self.decay
        self.target_metric = max(self.target_metric, self.floor)
        self.target_metric = min(self.target_metric, self.original_metric)

    def get_target(self):
        return self.target_metric
