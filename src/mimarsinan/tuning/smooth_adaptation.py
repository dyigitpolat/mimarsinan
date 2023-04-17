from mimarsinan.tuning.basic_interpolation import BasicInterpolation

class BasicSmoothAdaptation:
    def __init__(self, adaptation_function):
        self.adaptation_function = adaptation_function

    def adapt_smoothly(self, interpolators, cycles):
        for cycle in range(cycles):
            t = (cycle + 1) / cycles
            self.adaptation_function(*[i(t) for i in interpolators])
        
