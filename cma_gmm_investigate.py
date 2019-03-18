import numpy as np, os
import matplotlib.pyplot as plt
from gmm_gmr.mixtures import GMM_GMR
from lfd_improve.data import Demonstration
import matplotlib.patches as mpatches

data_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'

n_episode = 10
n_rollout = 5

demo = Demonstration(os.path.join(data_dir, '1'))
model = GMM_GMR(demo.ee_poses.reshape(1,-1,7), demo.times, n_clusters=4)
model.fit()

exp = 116
n_row = n_rollout//2
n_col = n_rollout//2 + n_rollout%2

coords = [1,2]
c1, c2 = coords
c1 -= 1
c2 -= 1

for e in range(1, n_episode+1):
    f, axs = plt.subplots(n_row, n_col)

    episode_dir = os.path.join(data_dir, 'ex{}'.format(exp), str(e))
    centers = np.loadtxt(os.path.join(episode_dir, '1', 'gmm_means.csv'), delimiter=',')
    covars =  np.load(os.path.join(episode_dir, '1', 'gmm_covs.csv.npy'))
    next_mean = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'gmm_means.csv'), delimiter=',')
    next_cov = np.load(os.path.join(episode_dir, str(n_rollout), 'gmm_covs.csv.npy'))

    n_center = len(centers)

    ts = centers[:,0]

    for row in range(n_row):
        for col in range(n_col):
            r = row * n_col + col

            if r == n_rollout:
                break

            exp_center = np.loadtxt(os.path.join(episode_dir, str(r+1), 'exp_w.csv'), delimiter=',')
            reward = np.sum(np.loadtxt(os.path.join(episode_dir, str(r+1), 'log.csv'), delimiter=',')[:2])

            t = np.array(ts).reshape(1,-1)
            c = np.concatenate([t, exp_center], axis=0)

            model.plot_gmm(axs[row][col], coords, means=c.T, covs=covars, cov_cent=centers)
            axs[row][col].set_title(str(reward))

    # Current trajectory
    model.plot_gmm(axs[-1][-1], coords, means=centers, covs=covars, clusters=False, alpha=0.5, linestyle=':')
    # Updated trajectory
    trj = model.plot_gmm(axs[-1][-1], coords, means=next_mean, covs=next_cov)

    # Set axis limits

    max_c1 = np.max(trj[:,c1])
    max_c2 = np.max(trj[:,c2])
    min_c1 = np.min(trj[:,c1])
    min_c2 = np.min(trj[:,c2])

    for (m, n), subplot in np.ndenumerate(axs):
        subplot.set_xlim(min_c1-0.01, max_c1+0.01)
        subplot.set_ylim(min_c2-0.01, max_c2+0.01)

    # patches = []
    # for i in range(n_center):
    #     patches.append(mpatches.Patch(color=colors[i], label=str(i)))
    #
    # plt.legend(handles=patches)
    f.suptitle('Episode {}'.format(e))
    plt.show()
