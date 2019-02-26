import numpy as np, copy
from lfd_improve.experiment import Experiment
from lfd_improve import learning
from lfd_improve.utils import isPD, nearestPD
import matplotlib.pyplot as plt

def P_r(k, K_e):
    beta = (1. + np.sum([np.log(i) for i in range(1,K_e+1)]))/float(K_e)
    alpha = np.exp(beta)/(K_e+1)

    return np.log(alpha*(K_e+1)) - np.log(k)

ex = Experiment('/home/ceteke/Desktop/iros_demos/open/ex2')
w_vars = np.diff(ex.weights, axis=0).mean(axis=-1)

std = 5.
initial_std = std

demo_dir = '/home/ceteke/Desktop/iros_demos/open/1'
to = learning.TrajectoryLearning(demo_dir, 20, 100, 5, 20, True)
to.dmp.w = ex.weights[0]
ex.weights = ex.weights[1:]

covs = np.array([np.eye(7)] * 20)
rews = []
exp_ws = []
K_e = 5

w = copy.deepcopy(to.dmp.w)

for e in range(20):
    exp_w = np.zeros_like(w)

    for i in range(len(covs)):
        exp_w[:,i] = np.random.multivariate_normal(w[:,i], np.square(std)*covs[i])

    exp_ws.append(exp_w)
    rews.append(ex.total_rewards[e])

    elite_idxs = np.argsort(rews)[::-1]
    upper = min(K_e, e+1)
    elite_idxs = elite_idxs[:upper]

    dW = np.array(exp_ws) - w
    rewards = np.array(rews)

    w += np.sum(dW[elite_idxs] * rewards[elite_idxs].reshape(-1,1,1), axis=0) / np.sum(rewards[elite_idxs], axis=0)

    if len(elite_idxs) > 1:
        for i in range(len(covs)):
            new_cov = np.zeros_like(covs[i])
            for k, e_idx in enumerate(elite_idxs):
                diff = (exp_ws[e_idx][:,i]-w[:,i]).reshape(-1,1)
                new_cov += P_r(k+1, len(elite_idxs)) * np.matmul(diff, diff.T)
            #covs[i] = new_cov
            new_cov = nearestPD(new_cov) if not isPD(new_cov) else new_cov
            covs[i] = new_cov
            #print covs[i]

    _, x, _, _ = to.dmp.imitate(w)
    plt.plot(x[:,0], x[:,1], label='Ep{}'.format(e))

    if e >= 5:
        std = initial_std * (1. - ((e - 5.) / (20. - 5.)))

plt.show()