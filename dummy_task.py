import pickle, time
import matplotlib.pyplot as plt
from dmp.rl import DMPPower, DMPCMA
from gmm_gmr.rl import GMMCMA
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import copy


class Spliner(object):
    def __init__(self, t, x):
        self.t = t
        self.splines = []
        for i in range(x.shape[-1]):
            self.splines.append(interpolate.UnivariateSpline(t, x[:,i], k=5))

    @property
    def get_motion(self):
        x = np.zeros((len(self.t), 7))
        dx = np.zeros_like(x)
        ddx = np.zeros_like(dx)
        dddx = np.zeros_like(ddx)

        for i in range(7):
            x[:,i] = self.splines[i](self.t)
            dx[:, i] = self.splines[i].derivative(1)(self.t)
            ddx[:, i] = self.splines[i].derivative(2)(self.t)
            dddx[:, i] = self.splines[i].derivative(3)(self.t)

        return self.t, x, dx, ddx, dddx

def jerk_reward(t, x):
    episode_spliner = Spliner(t, x)
    _, _, _, _, dddx_episode = episode_spliner.get_motion
    total_jerk = np.linalg.norm(dddx_episode, axis=1).sum()
    return 1. / total_jerk

def decay_std(initial, e, n_episode, decay_episode):
    if e >= decay_episode:
        return initial * (1. - (float(e - decay_episode) / (n_episode - decay_episode)))
    return initial

std = 50.
initial_std = std
n_episode = 50
decay_episode = 10
n_run = 5

robot_data = pickle.load(open('/home/ceteke/Desktop/sim_demos/1/robot_states.pk', 'rb'))[1:]
times = np.array([r[0] for r in robot_data])
ee_poses = np.array([r[4] for r in robot_data])
times = times - times[0]

print "DMP PoWER"
power_rewards = []
time.sleep(0.5)

for r in tqdm(range(n_run)):
    dmp = DMPPower(20,100,5)

    spliner = Spliner(times, ee_poses)
    times, y_gold, yd_gold, ydd_gold, yddd_gold = spliner.get_motion

    dmp.fit(times, y_gold, yd_gold, ydd_gold)

    rewards = []

    for e in range(n_episode):
        t_episode, x_episode,_, _ = dmp.generate_episode(std)
        reward = jerk_reward(t_episode, x_episode)
        dmp.update(reward)
        std = decay_std(initial_std, e+1, n_episode, decay_episode)

        t_greedy, x_greedy, _, _ = dmp.imitate()
        rewards.append(jerk_reward(t_greedy, x_greedy))

    power_rewards.append(rewards)

power_rewards = np.array(power_rewards)
r_mean = power_rewards.mean(axis=0)
r_var = power_rewards.std(axis=0)
x = range(1,n_episode+1)

plt.plot(x, r_mean, label='PoWER DMP')
plt.fill_between(x, r_mean+r_var, r_mean-r_var, alpha=0.5)

# ---- Adaptive Covariance Update --------

print "DMP CMA"

power_rewards = []
time.sleep(0.5)

n_offspring = 5

for r in tqdm(range(n_run)):

    dmp = DMPCMA(20, 100, n_offspring, std_init=10)

    spliner = Spliner(times, ee_poses)
    times, y_gold, yd_gold, ydd_gold, yddd_gold = spliner.get_motion

    dmp.fit(times, y_gold, yd_gold, ydd_gold)

    rewards = []

    for e in range(n_episode):
        for _ in range(n_offspring):
            t_episode, x_episode,_, _ = dmp.generate_episode()
            reward = jerk_reward(t_episode, x_episode)
            dmp.update(reward)

        t_greedy, x_greedy, _, _ = dmp.imitate()
        rewards.append(jerk_reward(t_greedy, x_greedy))
    power_rewards.append(rewards)

power_rewards = np.array(power_rewards)
r_mean = power_rewards.mean(axis=0)
r_var = power_rewards.std(axis=0)
x = range(1,n_episode+1)

plt.plot(x, r_mean, label='CMA DMP')
plt.fill_between(x, r_mean+r_var, r_mean-r_var, alpha=0.5)

print "GMM-GMR CMA"

all_rewards = []
time.sleep(0.5)

n_offspring = 5

for r in tqdm(range(n_run)):

    gmm = GMMCMA(np.array([ee_poses]), 0.5, n_offspring, times, n_clusters=5)
    gmm = copy.deepcopy(gmm)

    rewards = []

    for e in range(n_episode):
        generation_rewards = []
        for _ in range(n_offspring):
            t_episode, x_episode, _, _ = gmm.generate_episode()
            reward = jerk_reward(t_episode, x_episode)
            gmm.update(reward)

        t_greedy, x_greedy, _, _ = gmm.imitate()
        rewards.append(jerk_reward(t_greedy, x_greedy))
    all_rewards.append(rewards)

all_rewards = np.array(all_rewards)
r_mean = all_rewards.mean(axis=0)
r_var = all_rewards.std(axis=0)
x = range(1, n_episode + 1)

plt.plot(x, r_mean, label='CMA GMM-GMR')
plt.fill_between(x, r_mean + r_var, r_mean - r_var, alpha=0.5)

plt.legend()
plt.show()


