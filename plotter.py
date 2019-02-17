from lfd_improve.experiment import Experiment
from lfd_improve.data import Demonstration
from lfd_improve.spliner import Spliner
from lfd_improve.utils import get_jerk_reward
import numpy as np
import matplotlib.pyplot as plt

plot_greedy = True
skill_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'
demo_dir = '{}/1'.format(skill_dir)
beta = 25.6513944222

demo = Demonstration(demo_dir)
spliner = Spliner(demo.times, demo.ee_poses)
_,_,_,_,dddx = spliner.get_motion

baseline_jerk = beta  * get_jerk_reward(dddx)

experiment_idxs = range(11,16)

experiments = [Experiment('{}/ex{}'.format(skill_dir,e)) for e in experiment_idxs]
n_episode = len(experiments[0].episode_dirs)
n_greedy = len(experiments[0].greedy_dirs)

N = n_greedy if plot_greedy else n_episode

perception_rewards = np.zeros((len(experiments), N))
jerk_rewards = np.zeros((len(experiments), N))
total_rewards = np.zeros((len(experiments), N))

for i, exp in enumerate(experiments):
    perception_rewards[i] = exp.perception_rewards_greedy
    jerk_rewards[i] = exp.jerk_rewards_greedy

total_rewards = perception_rewards + jerk_rewards

total_std = total_rewards.var(axis=0)
perception_std = perception_rewards.var(axis=0)
jerk_std = jerk_rewards.var(axis=0)

total_mean = total_rewards.mean(axis=0)
perception_mean = perception_rewards.mean(axis=0)
jerk_mean = jerk_rewards.mean(axis=0)

plt.plot(total_mean, label='Total Reward')
plt.fill_between(range(N), total_mean-total_std, total_mean+total_std, alpha=0.5)
plt.plot(perception_mean, label='Perception Reward')
plt.fill_between(range(N), perception_mean-perception_std, perception_mean+perception_std, alpha=0.5)
plt.plot(jerk_mean, label='Jerk Reward')
plt.fill_between(range(N), jerk_mean-jerk_std, jerk_mean+jerk_std, alpha=0.5)
plt.axhline(baseline_jerk, label='Jerk Baseline', linestyle='--', c='black')
plt.legend()
plt.show()