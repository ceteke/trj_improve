from lfd_improve.experiment import Experiment
from lfd_improve.data import Demonstration
from lfd_improve.spliner import Spliner
from lfd_improve.utils import get_jerk_reward
import numpy as np
import matplotlib.pyplot as plt

plot_greedy = True
skill_dir = '/home/ceteke/Desktop/dmp_improve_demos/open'
demo_dir = '{}/1'.format(skill_dir)

demo = Demonstration(demo_dir)
spliner = Spliner(demo.times, demo.ee_poses)
_,_,_,_,dddx = spliner.get_motion

baseline_jerk = 0.5

experiment_names = {
    'PoWER Sparse': range(53,63),
    'PoWER Dense': range(63,73),
    'CMA DMP': range(73,83),
    'CMA GMM': range(83,93)
}

for experiment_name, experiment_idxs in experiment_names.items():
    freq = 6 if 'CMA' in experiment_name else None
    experiments = [Experiment('{}/ex{}'.format(skill_dir,e), freq=freq) for e in experiment_idxs]
    perception_greedy_all = np.concatenate([ex.perception_rewards_greedy for ex in experiments])
    jerk_greedy_all = np.concatenate([ex.jerk_rewards_greedy for ex in experiments])

    #np.savetxt('perception_all.csv', perception_greedy_all.reshape(len(experiment_idxs), -1), delimiter=',')
    #np.savetxt('jerk_all.csv', jerk_greedy_all.reshape(len(experiment_idxs), -1), delimiter=',')

    n_episode = len(experiments[0].episode_dirs)
    n_greedy = len(experiments[0].greedy_dirs)

    N = n_greedy if plot_greedy else n_episode

    perception_rewards = np.zeros((len(experiments), N))
    jerk_rewards = np.zeros((len(experiments), N))
    total_rewards = np.zeros((len(experiments), N))

    for i, exp in enumerate(experiments):
        perception_rewards[i] = exp.perception_rewards_greedy
        jerk_rewards[i] = exp.jerk_rewards_greedy
        #if 'CMA' in experiment_name:
        #    jerk_rewards[i] /= 2

    total_rewards = perception_rewards + jerk_rewards

    total_std = total_rewards.var(axis=0)
    perception_std = perception_rewards.var(axis=0)
    jerk_std = jerk_rewards.var(axis=0)

    total_mean = total_rewards.mean(axis=0)
    first_reward = total_mean[0]
    perception_mean = perception_rewards.mean(axis=0)
    jerk_mean = jerk_rewards.mean(axis=0)

    plt.plot(total_mean, label=experiment_name)
    plt.fill_between(range(N), total_mean-total_std, total_mean+total_std, alpha=0.2)
    #plt.plot(perception_mean, label=experiment_name)
    #plt.fill_between(range(N), perception_mean-perception_std, perception_mean+perception_std, alpha=0.1)
    #plt.plot(jerk_mean, label=experiment_name)
    #plt.fill_between(range(N), jerk_mean-jerk_std, jerk_mean+jerk_std, alpha=0.1)
plt.axhline(baseline_jerk, label='Jerk Baseline', linestyle='--', c='black')
plt.title("Total Reward vs. n^th Greedy")
plt.legend()
plt.show()

