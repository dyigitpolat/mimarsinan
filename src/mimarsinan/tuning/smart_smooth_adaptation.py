from mimarsinan.tuning.basic_smooth_adaptation import BasicSmoothAdaptation

class SmartSmoothAdaptation(BasicSmoothAdaptation):
    def __init__(
        self, adaptation_function, 
        state_clone_function, state_restore_function,
        evaluation_function):

        super().__init__(adaptation_function)
        self.state_clone_function = state_clone_function
        self.state_restore_function = state_restore_function
        self.evaluation_function = evaluation_function
        self.tolerance = 0.1

    def _find_step_size(self, t, interpolators):
        step_size = (1 - t)
        state = self.state_clone_function()
        prev_metric = self.evaluation_function(*[i(t) for i in interpolators])

        while current_metric < prev_metric * (1 - self.tolerance):
            next_t = t + step_size
            current_metric = self.evaluation_function(*[i(next_t) for i in interpolators])

            if current_metric < prev_metric * (1 - self.tolerance):
                self.state_restore_function(state)
                step_size /= 2

        return step_size

    def adapt_smoothly(self, interpolators, max_cycles):
        t = 0
        cycles = 0
        while t < 1 and cycles < max_cycles:
            step_size = self._find_step_size()
            t += step_size
            self.adaptation_function(*[i(t) for i in interpolators])

