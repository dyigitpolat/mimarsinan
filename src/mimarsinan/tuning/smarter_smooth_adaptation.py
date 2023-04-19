from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

class SmarterSmoothAdaptation(SmartSmoothAdaptation):
    def __init__(
        self, adaptation_function, 
        state_clone_function, state_restore_function,
        evaluation_function, target_metric):
        super().__init__(
            adaptation_function, 
            state_clone_function, state_restore_function,
            evaluation_function)
        
        self.best_metric = target_metric
        self.tolerance = (1.0 - target_metric) * 0.1 + 0.01

    def _lower_bound(self, t, interpolators):
        metric = self.evaluation_function(*[i(t) for i in interpolators])
        if metric > self.best_metric:
            self.best_metric = metric

        return self.best_metric * (1 - self.tolerance)

