class BasicInterpolation:
    def __init__(self, start, end, curve = lambda x: x):
        self.start = start
        self.end = end
        self.curve = curve

        error_eps = 1e-6
        assert abs(curve(0.0) - 0.0) < error_eps
        assert abs(curve(1.0) - 1.0) < error_eps
        assert curve(0.5) < 1.0
        assert curve(0.5) > 0.0

    def __call__(self, t):
        return self.curve(t) * (self.end - self.start) + self.start
    