from lfd_improve.learning import TrajectoryLearning
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os

experiment_dir = '/home/ceteke/Desktop/lfd_improve_demos/open/ex1'

n_episode = 5
n_rollout = 5

tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos/open', 15, 200, 5, 5, True)

ncol = 3
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
f.show()

for e in range(1,n_episode+1):
    episode_dir = os.path.join(experiment_dir, str(e))

    f1, axs1 = plt.subplots(5)
    f2, axs2 = plt.subplots(2)
    f3, axs3 = plt.subplots(1,2)

    for r in range(1,n_rollout+1):
        rollout_dir = os.path.join(episode_dir, str(r))
        x = np.loadtxt(os.path.join(rollout_dir, 'x.csv'), delimiter=',')
        axs1[r-1].plot(x[:,1], x[:,2])

    tl.dmp.w = np.loadtxt(os.path.join(episode_dir, '1', 'w.csv'), delimiter=',')
    C_curr = np.load(os.path.join(episode_dir, '1', 'cma_cov.csv.npy'))
    tl.dmp.update_C_cma(C_curr)
    std_curr = np.loadtxt(os.path.join(episode_dir, '1', 'sigma.csv'), delimiter=',')[:nrow*ncol]
    tl.dmp.sigmas = std_curr

    t, x, _, _ = tl.dmp.imitate()
    axs2[0].plot(x[:, 0], x[:, 1])

    for _ in range(25):
        t, x, _, _ = tl.dmp.generate_episode()
        axs2[0].plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

    tl.dmp.w = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'w.csv'), delimiter=',')
    C_next = np.load(os.path.join(episode_dir, str(n_rollout), 'cma_cov.csv.npy'))
    tl.dmp.update_C_cma(C_next)
    std_next = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'sigma.csv'), delimiter=',')[:nrow*ncol]
    tl.dmp.sigmas = std_next

    t, x, _, _ = tl.dmp.imitate()
    axs2[1].plot(x[:, 0], x[:, 1])

    for _ in range(25):
        t, x, _, _ = tl.dmp.generate_episode()
        axs2[1].plot(x[:, 0], x[:, 1], linestyle=':', alpha=0.5)

    f1.show()
    f2.show()

    cov_curr = C_curr * np.square(std_curr.reshape(-1,1,1))
    cov_next = C_next * np.square(std_next.reshape(-1,1,1))

    axs3[0].bar(range(nrow*ncol), [np.linalg.norm(c) for c in cov_curr])
    axs3[1].bar(range(nrow*ncol), [np.linalg.norm(c) for c in cov_next])

    f3.show()

    plt.show()