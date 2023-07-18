from mimarsinan.tuning.basic_interpolation import BasicInterpolation

class BasicSmoothAdaptation:
    def __init__(self, adaptation_function, interpolators):
        self.adaptation_function = adaptation_function
        self.interpolators = interpolators

    def adapt_smoothly(self, cycles):
        for cycle in range(cycles):
            t = (cycle + 1) / cycles
            self.adaptation_function(*[i(t) for i in self.interpolators])
