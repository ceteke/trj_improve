import numpy as np
from lfd_improve.experiment import Experiment
import matplotlib.pyplot as plt


skill_dir = '/home/ceteke/Desktop/lfd_improve_demos_sim/close'
experiments = list(range(212, 232))

success_greedy = []
reward_greedy = []

for idx in experiments:
    experiment = Experiment('{}/ex{}'.format(skill_dir, idx))

    reward_greedy.append(experiment.perception_rewards_greedy)
    success_greedy.append(experiment.successes_greedy)

success_greedy = np.array(success_greedy)
reward_greedy = np.array(reward_greedy)

f, axs = plt.subplots(1, 2)
X = list(range(1, 7))

success_var = success_greedy.var(axis=0)
success_mean = success_greedy.mean(axis=0)

axs[0].plot(X, success_mean)
axs[0].set_title("Success")
axs[0].fill_between(X, success_mean+success_var, success_mean-success_var, alpha=0.2)

reward_var = reward_greedy.std(axis=0)
reward_mean = reward_greedy.mean(axis=0)

axs[1].set_title("Reward")
axs[1].plot(X, reward_mean)

plt.suptitle("Open (n=5)")
plt.show()
