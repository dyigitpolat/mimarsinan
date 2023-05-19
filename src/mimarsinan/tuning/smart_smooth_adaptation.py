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
        self.tolerance = 0.01
        self.min_step = 0.0001

    def _adjust_minimum_step(self, step_size, t):
        if step_size < self.min_step:
            self.min_step *= 2.0

    def _lower_bound(self, t, interpolators):
        prev_metric = self.evaluation_function(*[i(t) for i in interpolators])
        return prev_metric * (1 - self.tolerance)

    def _find_step_size(self, t, interpolators):
        step_size = (1 - t)
        state = self.state_clone_function()

        current_metric = 0
        lower_bound = self._lower_bound(t, interpolators)
        while current_metric < lower_bound and step_size > self.min_step:
            print("step_size: ", step_size)
            next_t = t + step_size
            current_metric = self.evaluation_function(*[i(next_t) for i in interpolators])

            if current_metric < lower_bound:
                self.state_restore_function(state)
                step_size /= 2

        self._adjust_minimum_step(step_size, t)

        print("current_metric: ", current_metric)
        return step_size

    def adapt_smoothly(self, interpolators, max_cycles = None):
        t = 0
        cycles = 0
        while t < 1 and (not max_cycles or cycles < max_cycles):
            step_size = self._find_step_size(t, interpolators)
            t += step_size
            self.adaptation_function(*[i(t) for i in interpolators])
            cycles += 1

