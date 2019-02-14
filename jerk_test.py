import pickle, numpy as np
from lfd_improve.spliner import Spliner
from dmp.rl import DMPPower
import matplotlib.pyplot as plt


robot_data = pickle.load(open('/home/ceteke/Desktop/dmp_improve_demos/open/1/robot_states.pk', 'rb'))[1:]
times = np.array([r[0] for r in robot_data])
ee_poses = np.array([r[4] for r in robot_data])
times = times - times[0]

std = 78
initial = std

spl = Spliner(times, ee_poses)
ts, x, dx, ddx, dddx = spl.get_motion

dmp = DMPPower(20, 100, 5)
dmp.fit(ts, x, dx, ddx)

rewards = []
n_episode = 50
decay_episode = n_episode/2.

for e in range(1, n_episode+1):
    print std
    t_episode, x_episode, _, _ = dmp.generate_episode(std)
    spl_episode = Spliner(t_episode, x_episode)
    _, _, _, _, dddx_episode = spl_episode.get_motion

    jerk_reward = 1./np.linalg.norm(dddx_episode, axis=1).sum()

    if e >= decay_episode:
        std = initial * (1. - ((e - decay_episode) / (n_episode - decay_episode)))

    dmp.update(jerk_reward)
    rewards.append(jerk_reward)

plt.plot(rewards)
plt.show()