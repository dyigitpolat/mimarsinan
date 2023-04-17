class BasicInterpolation:
    def __init__(self, start, end, curve = lambda x: x):
        self.start = start
        self.end = end
        self.curve = curve

    def __call__(self, t):
        return self.curve(t) * (self.end - self.start) + self.start