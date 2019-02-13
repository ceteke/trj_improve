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
        x = np.zeros((len(self.t), 7))
        dx = np.zeros_like(x)
        ddx = np.zeros_like(dx)
        dddx = np.zeros_like(ddx)

        for i in range(7):
            x[:,i] = self.splines[i](self.t)
            dx[:, i] = self.splines[i].derivative(1)(self.t)
            ddx[:, i] = self.splines[i].derivative(2)(self.t)
            dddx[:, i] = self.splines[i].derivative(3)(self.t)

        return self.t, x, dx, ddx, dddx