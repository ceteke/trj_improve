from lfd_improve.experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


skill_dir = '/Volumes/Feyyaz/MSc/lfd_improve_demos/open2'
skill = 'Open'
is_reward = True

experiment_names = {
    'PI2-ES-Cov': list(range(1, 6))
}

for _, idxs_dense in experiment_names.items():
    experiments_dense = [Experiment('{}/ex{}'.format(skill_dir, i)) for i in idxs_dense]

    success_dense = np.array(map(lambda e: e.successes_greedy, experiments_dense))
    dense_mean = np.mean(success_dense, axis=0)
    dense_var = np.var(success_dense, axis=0)

    reward_dense = np.array(map(lambda e: e.perception_rewards_greedy, experiments_dense))
    reward_mean = np.mean(reward_dense, axis=0)
    rewad_var = np.std(reward_dense, axis=0)


    plt.title(skill)

    X = list(range(1, len(dense_mean)+1))



    if is_reward:
        plt.plot(X, reward_mean, marker='1', markersize=16)
        #plt.fill_between(X, reward_mean + rewad_var, reward_mean - rewad_var, alpha=0.2)

    else:
        plt.plot(X, dense_mean, marker='1', markersize=16)
        plt.fill_between(X, np.clip(dense_mean + dense_var, 0, 1), np.clip(dense_mean - dense_var, 0, 1), alpha=0.2)

    if not is_reward:
        plt.ylim((0, 1.01))
        plt.xlabel('Greedy')
        plt.ylabel('Success')
    else:
        plt.xlabel('Greedy')
        plt.ylabel('Return')
    #plt.legend()
    plt.savefig('/Users/cem/Desktop/{}.png'.format(skill.lower() + '_rew' if is_reward else ''), bbox_inches="tight", dpi=400)
    plt.show()