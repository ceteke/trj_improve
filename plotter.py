from lfd_improve.experiment import Experiment
from lfd_improve.data import Demonstration
from lfd_improve.spliner import Spliner
from lfd_improve.utils import get_jerk_reward
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

plot_greedy = True
skill_dir = '/home/ceteke/Desktop/lfd_improve_demos/open2'
demo_dir = '{}/1'.format(skill_dir)

demo = Demonstration(demo_dir)
spliner = Spliner(demo.times, demo.ee_poses)
_,_,_,_,dddx = spliner.get_motion

baseline_jerk = 0.5

experiment_names = {
    #'PoWER Sparse': range(53,63),
    #'PoWER Dense': range(63,73),
    #'Sparse': range(254,274),
    #'PoWER Dense': range(194,214),
    'Dense': range(1,4)
}

data = {}

f, axs = plt.subplots(1,len(experiment_names))

if len(experiment_names) == 1:
    axs = [axs]

for axidx, (experiment_name, experiment_idxs) in enumerate(experiment_names.items()):
    experiments = [Experiment('{}/ex{}'.format(skill_dir,e)) for e in experiment_idxs]
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
        perception_rewards[i] = exp.successes_greedy
        jerk_rewards[i] = exp.jerk_rewards_greedy
        #if 'CMA' in experiment_name:
        #    jerk_rewards[i] /= 2

    #print perception_rewards[:,-1]
    #print jerk_rewards[:,-1]

    total_rewards = perception_rewards #+ jerk_rewards

    data[experiment_name] = total_rewards

    total_std = total_rewards.std(axis=0)
    total_mean = np.mean(total_rewards, axis=0)

    axs[axidx].plot(range(1,len(total_mean)+1), total_mean, label=experiment_name, marker='o', linestyle=':')
    axs[axidx].set_title(experiment_name)

    #plt.boxplot(total_rewards)
    #plt.plot(range(1, N+1), total_mean-total_std, alpha=0.9, color=c, linestyle=':')
    #plt.plot(range(1, N+1), total_mean + total_std, alpha=0.9, color=c, linestyle=':')
    #plt.errorbar(range(1, N+1), total_mean, yerr=total_std, fmt='o',
    #             elinewidth=1, capsize=10, label=experiment_name)
    axs[axidx].boxplot(perception_rewards)
    #plt.plot(perception_mean, label=experiment_name)
    #plt.fill_between(range(1,len(total_mean)+1), total_mean-total_std, total_mean+total_std, alpha=0.1)
    #plt.plot(jerk_mean, label=experiment_name)
    #plt.fill_between(range(N), jerk_mean-jerk_std, jerk_mean+jerk_std, alpha=0.1)
    axs[axidx].set_xlabel('Episode')
    axs[axidx].set_ylabel('Success')
    axs[axidx].grid()
#plt.axhline(baseline_jerk, label='Jerk Baseline', linestyle='--', c='black')

#plt.title("Success")
#plt.legend()
plt.suptitle("Open (Real, n=3)")

#plt.savefig('/home/ceteke/Desktop/box_dense.png')
plt.show()

