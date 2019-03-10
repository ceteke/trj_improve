import os, numpy as np


class Experiment(object):
    def __init__(self, ex_dir, freq=None):
        self.ex_dir = ex_dir
        self.episode_dirs = list(
            map(lambda x: os.path.join(ex_dir, x),
                filter(lambda x: str.isdigit(x), os.listdir(self.ex_dir))))
        self.episode_dirs = sorted(self.episode_dirs,
                                   key=lambda x: int(os.path.basename(os.path.normpath(x))))

        #self.perception_rewards, self.jerk_rewards, self.successes, self.variances = self.get_rewards(self.episode_dirs)
        #self.total_rewards = self.perception_rewards + self.jerk_rewards

        self.greedy_dirs = list(
            map(lambda x: os.path.join(ex_dir, x),
                filter(lambda x: 'greedy' in x, os.listdir(self.ex_dir))))
        self.greedy_dirs = sorted(self.greedy_dirs,
                                  key=lambda x: int(x.split('_')[-1]))

        if freq:
            idxs = np.arange(0, len(self.greedy_dirs), freq, dtype=np.int)
            self.greedy_dirs = np.array(self.greedy_dirs)[idxs]


        self.perception_rewards_greedy, self.jerk_rewards_greedy, self.successes_greedy, _ = self.get_rewards(self.greedy_dirs)

        #self.weights = np.array([np.loadtxt(os.path.join(e, 'dmp.csv'),
        #                                    delimiter=',') for e in [self.ex_dir] + self.episode_dirs])

    def get_rewards(self, dirs):
        jerk_rewards = np.zeros(len(dirs))
        perception_rewards = np.zeros(len(dirs))
        successes = np.zeros(len(dirs))
        variances = np.zeros(len(dirs))

        for i, ep_dir in enumerate(dirs):
            log_dir = os.path.join(ep_dir, 'log.csv')
            rewards = np.loadtxt(log_dir, delimiter=',')
            if len(rewards) > 3:
                perception_rewards[i], jerk_rewards[i], successes[i], variances[i] = rewards
            else:
                perception_rewards[i], jerk_rewards[i], successes[i] = rewards

        return perception_rewards, jerk_rewards, successes, variances

    def save_data(self):
        np.savetxt('perception_greedy.csv', self.perception_rewards_greedy, delimiter=',')
        np.savetxt('jerk_greedy.csv', self.jerk_rewards_greedy, delimiter=',')