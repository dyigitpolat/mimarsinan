import math


def target_decay_from_validation_samples(n_samples: int) -> float:
    """Decay factor from validation set size: ``1 - max(1/√n, 0.001)``, clamped to [0.95, 0.999]."""
    d = 1.0 - max(1.0 / math.sqrt(max(1, int(n_samples))), 0.001)
    return max(0.95, min(0.999, d))


class AdaptationTargetAdjuster:
    def __init__(self, original_target, decay=0.999, floor_ratio=0.90):
        assert decay < 1.0
        assert decay > 0.5
        assert 0.0 < floor_ratio <= 1.0

        self.decay = decay
        self.growth = 1.0 / decay
        self.target_metric = original_target
        self.original_metric = original_target
        self.floor = original_target * floor_ratio

    @classmethod
    def from_pipeline(cls, original_target, pipeline):
        """Build adjuster using validation-set-sized decay and pipeline-configured floor ratio."""
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

        dp = DataLoaderFactory(pipeline.data_provider_factory).create_data_provider()
        n = dp.get_validation_set_size()
        decay = target_decay_from_validation_samples(n)
        floor_ratio = float(pipeline.config.get("tuner_target_floor_ratio", 0.90))
        return cls(original_target, decay, floor_ratio)

    def update_target(self, new_metric):
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
