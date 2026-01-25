class VarianceMeter:
    """Tracks mean and standard deviation of a value."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.sum_sq = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.sum_sq += (val * val) * n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

    @property
    def std(self):
        if self.count < 2:
            return 0.0
        mean = self.avg
        # Var = E[x^2] - (E[x])^2
        variance = (self.sum_sq / self.count) - (mean ** 2)
        # Clamp to 0 to avoid numerical errors resulting in negative variance
        return (max(0.0, variance)) ** 0.5

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "} ± {std" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg, std=self.std)
