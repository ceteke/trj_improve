from lfd_improve.utils import normal_overlap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def plot_gauissian(mu, c, w):
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    rv = multivariate_normal(mu, c)
    results = w*rv.pdf(pos)

    plt.plot(y, results[0,:])

m1 = np.array([0,0])
m2 = np.array([0.5,0.5])

c1 = np.array([[1,0], [0,1]])
c2 = np.array([[1,0], [0,1]])

print normal_overlap(m1, c1, m2, c2, 0.5, 0.5)

fig = plt.figure()
plot_gauissian(m1, c1, 0.5)
plot_gauissian(m2, c2, 0.5)

plt.show()