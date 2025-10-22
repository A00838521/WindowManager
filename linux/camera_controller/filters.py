import math
from typing import Optional


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LowPass:
    def __init__(self, alpha: float, init: Optional[float] = None):
        self.a = alpha
        self.y = init
        self.s = False

    def filt(self, x: float, alpha: float) -> float:
        self.a = alpha
        if not self.s:
            self.y = x
            self.s = True
        else:
            self.y = self.y + alpha * (x - self.y)
        return self.y


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = LowPass(1.0)
        self.dx_prev = LowPass(1.0)
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / max(1e-6, dt))

    def filter(self, t: float, x: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev.filt(x, 1.0)
            self.dx_prev.filt(0.0, 1.0)
            return x
        dt = max(1e-6, t - self.t_prev)
        dx = (x - self.x_prev.y) / dt
        ad = OneEuro._alpha(self.d_cutoff, dt)
        edx = self.dx_prev.filt(dx, ad)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = OneEuro._alpha(cutoff, dt)
        ex = self.x_prev.filt(x, a)
        self.t_prev = t
        return ex
