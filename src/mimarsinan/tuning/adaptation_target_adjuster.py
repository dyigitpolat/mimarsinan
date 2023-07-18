class AdaptationTargetAdjuster:
    def __init__(self, original_target, decay = 0.999):
        assert decay < 1.0
        assert decay > 0.5

        self.decay = decay
        self.growth = 1.0 / decay
        self.target_metric = original_target
        self.original_metric = original_target

    def update_target(self, new_metric):
        if new_metric > self.target_metric:
            self.target_metric *= self.growth
        else:
            self.target_metric *= self.decay

    def get_target(self):
        return self.target_metric