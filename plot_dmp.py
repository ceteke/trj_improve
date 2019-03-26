from lfd_improve.learning import TrajectoryLearning
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

experiment_dir = '/home/ceteke/Desktop/lfd_improve_demos/open/ex1'

n_episode = 5
n_rollout = 5

tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos/open', 10, 200, 5, 5, True)
ncol = 2
nrow = 5

old_sigmas = deepcopy(tl.dmp.sigmas)
tl.dmp.sigmas = np.zeros(len(old_sigmas))

f, axs = plt.subplots(nrow, ncol)

for i in range(nrow):
    for j in range(ncol):
        s_idx = i*ncol + j
        tl.dmp.sigmas[s_idx] = old_sigmas[s_idx]
        axs[i][j].plot(tl.x_imitate[:, 0], tl.x_imitate[:, 1], label='imitate', color='black')

        for _ in range(10):
            t, x, _, _ = tl.dmp.generate_episode()
            axs[i,j].plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

        tl.dmp.sigmas = np.zeros(len(old_sigmas))
plt.show()