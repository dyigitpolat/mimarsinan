class BasicInterpolation:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.curve = lambda x: x

    def interpolate(self, t):
        return self.curve(t) * (self.end - self.start) + self.start