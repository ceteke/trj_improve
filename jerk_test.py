from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np


to = TrajectoryLearning('/home/ceteke/Desktop/sim_demos/1', 20, 100, 5)

ts, x, dx, ddx = to.generate_episode()

dddx = np.diff(ddx, axis=0)/to.delta
ts, x_d, dx_d, ddx_d, dddx_d = to.spliner.get_motion

plt.plot(ts, dddx_d[:,:3])
plt.plot(ts[:-1], dddx[:,:3], linestyle=':')
plt.show()