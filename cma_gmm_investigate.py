import numpy as np, os
import matplotlib.pyplot as plt
from gmm_gmr.mixtures import GMM_GMR
from lfd_improve.data import Demonstration
import copy
from gmm_gmr.utils import kl_gmm
from lfd_improve.utils import update_gmm


data_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'

n_episode = 10
n_rollout = 5

demo = Demonstration(os.path.join(data_dir, '1'))
model = GMM_GMR(demo.ee_poses.reshape(1,-1,7), demo.times, n_clusters=4)
model.fit()

exp = 83
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

    n_center = len(centers)

    covars =  np.load(os.path.join(episode_dir, '1', 'gmm_covs.csv.npy'))
    try:
        weights = np.loadtxt(os.path.join(episode_dir, '1', 'gmm_weights.csv'), delimiter=',')
    except:
        weights = np.array([1./n_center]*n_center)

    next_mean = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'gmm_means.csv'), delimiter=',')
    next_cov = np.load(os.path.join(episode_dir, str(n_rollout), 'gmm_covs.csv.npy'))
    n_center_next = len(next_mean)

    try:
        next_w = np.loadtxt(os.path.join(episode_dir, str(n_rollout), 'gmm_weights.csv'), delimiter=',')
    except:
        next_w = np.array([1./n_center_next]*n_center_next)

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

    patches = model.get_color_patches(len(centers))
    plt.legend(handles=patches)


    gmm_prev = copy.deepcopy(model.gmm)
    gmm_new = copy.deepcopy(model.gmm)

    update_gmm(gmm_prev, centers, covars, weights)
    update_gmm(gmm_new, next_mean, next_cov, next_w)

    model.gmm = copy.deepcopy(gmm_prev)
    bs = model.get_baseline_distance()

    model.gmm = copy.deepcopy(gmm_new)

    print "Baseline", bs, "New baseline", model.get_baseline_distance()
    to_merge = model.merge_clusters(threshold=0.1)

    print to_merge

    trj = model.plot_gmm(axs[-1][-1], coords, means=model.gmm.means_, covs=model.gmm.covariances_)

    if len(to_merge) > 0:
        print "Calculating kl"
        print kl_gmm(gmm_new, model.gmm)

    # Set axis limits

    max_c1 = np.max(trj[:, c1])
    max_c2 = np.max(trj[:, c2])
    min_c1 = np.min(trj[:, c1])
    min_c2 = np.min(trj[:, c2])

    for (m, n), subplot in np.ndenumerate(axs):
        subplot.set_xlim(min_c1 - 0.01, max_c1 + 0.01)
        subplot.set_ylim(min_c2 - 0.01, max_c2 + 0.01)

    f.suptitle('Episode {}'.format(e))
    plt.show()
