from mimarsinan.tuning.basic_smooth_adaptation import BasicSmoothAdaptation
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster

class SmartSmoothAdaptation(BasicSmoothAdaptation):
    def __init__(
        self, adaptation_function, 
        state_clone_function, state_restore_function,
        evaluation_function, interpolators):

        super().__init__(adaptation_function, interpolators)
        self.state_clone_function = state_clone_function
        self.state_restore_function = state_restore_function
        self.evaluation_function = evaluation_function
        self.min_step = 0.01
        self.tolerance = 0.01

        self.original_target = \
            self.evaluation_function(*[i(0) for i in self.interpolators])
        self.target_adjuster = AdaptationTargetAdjuster(self.original_target)

    def _adjust_minimum_step(self, step_size, t):
        halfway = (1 - t) / 2
        step_size_is_too_small = step_size < self.min_step
        min_step_is_within_halfway = self.min_step < halfway

        if step_size_is_too_small and min_step_is_within_halfway:
            self.min_step *= 2.0
            self.tolerance *= 1.5

    def _find_step_size(self, t):
        step_size = (1 - t) * 2
        state = self.state_clone_function()

        current_metric = 0
        tolerable_metric = self.target_adjuster.get_target() * (1.0 - self.tolerance)
        while current_metric < tolerable_metric and step_size > self.min_step:
            step_size /= 2
            print("step_size: ", step_size)
            
            next_t = t + step_size
            current_metric = self.evaluation_function(*[i(next_t) for i in self.interpolators])
            self.state_restore_function(state)

        self._adjust_minimum_step(step_size, t)

        print("current_metric: ", current_metric)
        return step_size

    def adapt_smoothly(self, max_cycles = None):
        t = 0
        cycles = 0
        while t < 1 and (not max_cycles or cycles < max_cycles):
            step_size = self._find_step_size(t)
            t += step_size
            interpolated_params = [i(t) for i in self.interpolators]
            self.adaptation_function(*interpolated_params)
            self.target_adjuster.update_target(self.evaluation_function(*interpolated_params))
            cycles += 1

