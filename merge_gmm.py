import numpy as np, os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
from random import shuffle
from gmm_gmr.mixtures import GMM_GMR
from lfd_improve.data import Demonstration
from lfd_improve.utils import center_distance
import matplotlib.patches as mpatches


def plot_ellipse(ax, pos, cov, color):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 4 * np.sqrt(vals)
    ellip = mpl.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, lw=1, fill=True, alpha=0.2, color=color)

    ax.add_artist(ellip)

colors = mcolors.get_named_colors_mapping().keys()
shuffle(colors)

data_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'
ex_ids = list(range(83,93))
n_episode = 10
n_rollout = 5
n_center = 6
demo = Demonstration(os.path.join(data_dir, '1'))
model = GMM_GMR(demo.ee_poses.reshape(1,-1,7), demo.times, n_clusters=n_center)
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
n_row = n_rollout//2
n_col = n_rollout//2 + n_rollout%2

coords = [1,2]

c1, c2 = coords
upper = coords[-1] + 1

for e in range(1, n_episode+1):
    f, axs = plt.subplots(n_row, n_col)
    centers = means[exp, e-1]
    covars = covs[exp, e-1]
    exp_centers = explored_means[exp, e-1]
    ts = centers[:,0]

    overlaps = np.zeros((n_center, n_center))

    for i in range(n_center):
        for j in range(n_center):
            overlaps[i,j] = center_distance(centers[i,1:], centers[j,1:])

    for row in range(n_row):
        for col in range(n_col):
            r = row * n_col + col

            if r == n_rollout:
                break

            for i in range(n_center):
                pos = exp_centers[r,:upper,i]
                cov = covars[i, c2:upper+1, c2:upper+1]
                axs[row][col].scatter(pos[c1], pos[c2], color=colors[i]) # Scatter exploration centers
                plot_ellipse(axs[row][col], centers[i,c2:upper+1], cov, colors[i]) # Plot current covs

            t = np.array(ts).reshape(1,-1)
            c = np.concatenate([t, exp_centers[r]], axis=0)

            _, trj = model.generate_trajectory(means=c.T, covs=covars)
            axs[row][col].plot(trj[:, c1], trj[:, c2], linestyle=':', color='black', alpha=0.6)
            axs[row][col].set_title(str(rewards[exp][e-1][r]))

    # Current trajectory
    _, trj = model.generate_trajectory(means=centers, covs=covars)
    axs[-1][-1].plot(trj[:, c1], trj[:, c2], color='black', linestyle=':', alpha=0.5)
    # Updated trajectory
    centers_updated = next_means[exp, e - 1]
    covars_updated = next_covs[exp, e - 1]

    _, trj = model.generate_trajectory(means=centers_updated, covs=covars_updated)
    axs[-1][-1].plot(trj[:, c1], trj[:, c2], color='black')

    # Set axis limits
    max_c1 = np.max(trj[:,c1])
    max_c2 = np.max(trj[:,c2])
    min_c1 = np.min(trj[:,c1])
    min_c2 = np.min(trj[:,c2])

    for (m, n), subplot in np.ndenumerate(axs):
        subplot.set_xlim(min_c1-0.01, max_c1+0.01)
        subplot.set_ylim(min_c2-0.01, max_c2+0.01)

    print np.min(overlaps[np.nonzero(overlaps)])

    patches = []
    for i in range(n_center):
        patches.append(mpatches.Patch(color=colors[i], label=str(i)))

    plt.legend(handles=patches)
    f.suptitle('Episode {}'.format(e))
    #plt.show()
