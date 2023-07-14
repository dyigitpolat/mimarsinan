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
        self.min_step = 0.0001

        self.tolerance = 0.01
        self.target_decay = 0.999
        self.target_metric = 1.0

    def _adjust_minimum_step(self, step_size, t):
        if step_size < self.min_step:
            self.min_step *= 2.0

    def _update_target(self, current_metric):
        self.target_metric = max(self.target_metric * self.target_decay, current_metric)

    def _find_step_size(self, t, interpolators):
        step_size = (1 - t) * 2
        state = self.state_clone_function()

        current_metric = 0
        while current_metric < self.target_metric and step_size > self.min_step:
            step_size /= 2
            print("step_size: ", step_size)
            
            next_t = t + step_size
            current_metric = self.evaluation_function(*[i(next_t) for i in interpolators])
            self._update_target(current_metric)
            self.state_restore_function(state)

        self._adjust_minimum_step(step_size, t)

        print("current_metric: ", current_metric)
        return step_size

    def adapt_smoothly(self, interpolators, max_cycles = None):
        state = self.state_clone_function()
        self.target_metric = \
            self.evaluation_function(*[i(0) for i in interpolators]) * (1 - self.tolerance)
        self.state_restore_function(state)

        t = 0
        cycles = 0
        while t < 1 and (not max_cycles or cycles < max_cycles):
            step_size = self._find_step_size(t, interpolators)
            t += step_size
            self.adaptation_function(*[i(t) for i in interpolators])
            cycles += 1

