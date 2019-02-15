from lfd_improve.experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt


experiment_idxs = range(1,11)

experiments = [Experiment('/home/ceteke/Desktop/dmp_improve_demos/open/ex{}'.format(e)) for e in experiment_idxs]
n_episode = len(experiments[0].episode_dirs)

perception_rewards = np.zeros((len(experiments), n_episode))
jerk_rewards = np.zeros((len(experiments), n_episode))
total_rewards = np.zeros((len(experiments), n_episode))

for i, exp in enumerate(experiments):
    perception_rewards[i] = exp.perception_rewards
    jerk_rewards[i] = exp.jerk_rewards

total_rewards = perception_rewards + jerk_rewards

total_std = total_rewards.var(axis=0)
perception_std = perception_rewards.var(axis=0)
jerk_std = jerk_rewards.var(axis=0)

total_mean = total_rewards.mean(axis=0)
perception_mean = perception_rewards.mean(axis=0)
jerk_mean = jerk_rewards.mean(axis=0)

plt.plot(total_mean)
plt.fill_between(range(n_episode), total_mean-total_std, total_mean+total_std, alpha=0.5)
plt.plot(perception_mean)
plt.fill_between(range(n_episode), perception_mean-perception_std, perception_mean+perception_std, alpha=0.5)
plt.plot(jerk_mean)
plt.fill_between(range(n_episode), jerk_mean-jerk_std, jerk_mean+jerk_std, alpha=0.5)
plt.show()