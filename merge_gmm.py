import numpy as np, os
import matplotlib.pyplot as plt
from gmm_gmr.mixtures import GMM_GMR
from lfd_improve.data import Demonstration
from lfd_improve.utils import center_distance


data_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'
ex_ids = list(range(83,93))
n_episode = 10
n_rollout = 5
n_center = 6

demo = Demonstration(os.path.join(data_dir, '1'))
model = GMM_GMR(demo.ee_poses.reshape(1,-1,7), demo.times, n_clusters=6)
model.fit()

means = np.zeros((len(ex_ids), n_episode, n_center, 8))
explored_means = np.zeros((len(ex_ids), n_episode, n_rollout, 7, n_center))
covs = np.zeros((len(ex_ids), n_episode, n_center, 8, 8))
rewards = np.zeros((len(ex_ids), n_episode, n_rollout))

next_means = np.zeros((len(ex_ids), n_episode, n_center, 8))
next_covs = np.zeros((len(ex_ids), n_episode, n_center, 8, 8))

experiment_dirs = [os.path.join(data_dir, 'ex{}'.format(i)) for i in ex_ids]

for i, ex in enumerate(experiment_dirs):
    for ep in range(1, n_episode+1):
        episode_dir = os.path.join(ex, str(ep))
        means[i, ep-1] = np.loadtxt(os.path.join(episode_dir, '1', 'gmm_means.csv'), delimiter=',')
        covs[i, ep-1] = np.load(os.path.join(episode_dir, '1', 'gmm_covs.csv.npy'))

        next_means[i, ep - 1] = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'gmm_means.csv'), delimiter=',')
        next_covs[i, ep - 1] = np.load(os.path.join(episode_dir, str(n_rollout), 'gmm_covs.csv.npy'))

        for rol in range(1, n_rollout+1):
            explored_means[i, ep-1, rol-1] = np.loadtxt(os.path.join(episode_dir, str(rol), 'exp_w.csv'), delimiter=',')
            rewards[i, ep-1, rol-1] = np.sum(np.loadtxt(os.path.join(episode_dir, str(rol), 'log.csv'), delimiter=',')[:2])

exp = 0
centers = means[exp, 0]
next_centers = means[exp, 1]
covars = covs[exp, 0]
next_covs = covs[exp, 1]
coords = [1,2,3]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
model.plot_gmm(ax, coords)

model.merge_clusters()

ax = fig.add_subplot(1, 2, 2, projection='3d')
model.plot_gmm(ax, coords)

plt.show()
