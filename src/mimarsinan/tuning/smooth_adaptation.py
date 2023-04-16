from mimarsinan.tuning.basic_interpolation import BasicInterpolation

class BasicSmoothAdaptation:
    def __init__(self, adaptation_function):
        self.adaptation_function = adaptation_function

    def adapt_smoothly(self, interpolation_ranges, cycles):
        interpolations = \
            [BasicInterpolation(*range) for range in interpolation_ranges]
        
        for cycle in range(cycles):
            t = (cycle + 1) / cycles
            interpolated_values = \
                [interpolation.interpolate(t) for interpolation in interpolations]
            self.adaptation_function(*interpolated_values)
        
