import numpy as np, os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
from random import shuffle
from gmm_gmr.mixtures import GMM_GMR
from lfd_improve.data import Demonstration


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

colors = mcolors.get_named_colors_mapping().values()
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

experiment_dirs = [os.path.join(data_dir, 'ex{}'.format(i)) for i in ex_ids]

for i, ex in enumerate(experiment_dirs):
    for ep in range(1, n_episode+1):
        episode_dir = os.path.join(ex, str(ep))
        means[i, ep-1] = np.loadtxt(os.path.join(episode_dir, '1', 'gmm_means.csv'), delimiter=',')
        covs[i, ep-1] = np.load(os.path.join(episode_dir, '1', 'gmm_covs.csv.npy'))

        for rol in range(1, n_rollout+1):
            explored_means[i, ep-1, rol-1] = np.loadtxt(os.path.join(episode_dir, str(rol), 'exp_w.csv'), delimiter=',')

exp = 0
n_row = n_rollout//2
n_col = n_rollout//2 + n_rollout%2


for e in range(1, n_episode+1):
    f, axs = plt.subplots(n_row, n_col)
    centers = means[exp, e-1]
    covars = covs[exp, e-1]
    exp_centers = explored_means[exp, e-1]
    ts = centers[:,0]

    for c in centers:
        c = c.reshape(1,-1)
        dists = np.linalg.norm(centers-c, axis=1)
        print np.mean(dists)
    print "=="

    for row in range(n_row):
        for col in range(n_col):
            r = row * n_col + col

            if r == n_rollout:
                break

            for i in range(n_center):
                pos = exp_centers[r,:2,i]
                cov = covars[i, 1:3, 1:3]
                axs[row][col].scatter(pos[0], pos[1], color=colors[i])
                plot_ellipse(axs[row][col], centers[i,1:3], cov, colors[i])

            t = np.array(ts).reshape(1,-1)
            c = np.concatenate([t, exp_centers[r]], axis=0)

            _, trj = model.generate_trajectory(means=c.T, covs=covars)
            axs[row][col].plot(trj[:, 0], trj[:, 1], linestyle=':', color='black', alpha=0.6)

    _, trj = model.generate_trajectory(means=centers, covs=covars)
    axs[-1][-1].plot(trj[:, 0], trj[:, 1], linestyle=':', color='black')
    plt.legend()
    #plt.show()
