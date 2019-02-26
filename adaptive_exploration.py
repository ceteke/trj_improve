from lfd_improve.utils import sampling_fn
import numpy as np
import matplotlib.pyplot as plt
from lfd_improve.experiment import Experiment
from lfd_improve import learning

# rew = np.arange(0,1,0.01)
# hs = [0.25, 0.5, 0.75, 1., 1.5, 1.75, 2., 2.5, 3.]
#
# y = sampling_fn(1-rew, 0, 15, .75)
# plt.plot(rew, y)
#
# plt.show()

ex = Experiment('/home/ceteke/Desktop/iros_demos/open/ex2')
w_vars = np.diff(ex.weights, axis=0).mean(axis=-1)


# plt.plot(w_vars[:,0])
# plt.plot(w_vars[:,1])
# plt.show()

std = 50

to = learning.TrajectoryLearning('/home/ceteke/Desktop/iros_demos/open/1', 20, 100, 5, 20, True)
to.dmp.w = ex.weights[0]

# _, x_init, _, _ = to.dmp.imitate()
# plt.plot(x_init[:,0], x_init[:,1], linestyle=':')

def P_r(k, K_e):
    beta = (1. + np.sum([np.log(i) for i in range(1,K_e+1)]))/float(K_e)
    alpha = np.exp(beta)/(K_e+1)

    return np.log(alpha*(K_e+1)) - np.log(k)

K = 20
K_e = 5

ps = []
for k in range(1,K_e+1):
    ps.append(P_r(k, K_e))

print np.sum(ps)

plt.plot(ps)
plt.show()
exit()

plt.figure(1)
plt.title("Zero mean")
for e in range(5):
    xs = []

    for _ in range(10):
        exp_w = ex.weights[e] + np.random.randn(*to.dmp.w.shape) * ex.variances[e]
        t, x, dx, ddx = to.dmp.imitate(exp_w)
        xs.append(x)

    xs = np.array(xs)
    xs_m = xs.mean(axis=0)

    plt.plot(xs_m[:,0], xs_m[:,1], label='Ep{}'.format(e))

plt.figure(2)
plt.title("Exploration Adaption")
for e in range(5):
    xs = []
    eps = np.linalg.norm(to.dmp.w, axis=0).mean()
    for _ in range(10):
        high_act = np.where(np.linalg.norm(ex.weights[e], axis=0) > eps)[0]
        mask = np.zeros_like(to.dmp.w)
        mask[:, high_act] = 1.0
        exp_w = ex.weights[e] + (w_vars[e].reshape(-1,1) + np.random.randn(*to.dmp.w.shape) * ex.variances[e]) * mask
        t, x, dx, ddx = to.dmp.imitate(exp_w)
        xs.append(x)

    xs = np.array(xs)
    xs_m = xs.mean(axis=0)

    plt.plot(xs_m[:,0], xs_m[:,1], label='Ep{}'.format(e))

plt.show()
