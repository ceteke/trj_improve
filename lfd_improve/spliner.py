import numpy as np
from scipy import interpolate


class Spliner(object):
    def __init__(self, t, x):
        self.t = t
        self.splines = []
        for i in range(x.shape[-1]):
            self.splines.append(interpolate.UnivariateSpline(t, x[:,i], k=5))

    @property
    def get_motion(self):
        dt = 0.05
        times = np.arange(self.t[0], self.t[-1]+dt, dt)

        x = np.zeros((len(times), 7))
        dx = np.zeros_like(x)
        ddx = np.zeros_like(dx)
        dddx = np.zeros_like(ddx)

        for i in range(7):
            x[:,i] = self.splines[i](times)
            dx[:, i] = self.splines[i].derivative(1)(times)
            ddx[:, i] = self.splines[i].derivative(2)(times)
            dddx[:, i] = self.splines[i].derivative(3)(times)

        return times, x, dx, ddx, dddx