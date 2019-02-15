import os, numpy as np


class Experiment(object):
    def __init__(self, ex_dir):
        self.ex_dir = ex_dir
        self.episode_dirs = list(
            map(lambda x: os.path.join(ex_dir, x),
                filter(lambda x: str.isdigit(x), os.listdir(self.ex_dir))))
        self.episode_dirs = sorted(self.episode_dirs,
                                   key=lambda x: int(os.path.basename(os.path.normpath(x))))

        self.jerk_rewards = np.zeros(len(self.episode_dirs))
        self.perception_rewards = np.zeros(len(self.episode_dirs))

        for i, ep_dir in enumerate(self.episode_dirs):
            log_dir = os.path.join(ep_dir, 'log.csv')
            rewards = np.loadtxt(log_dir, delimiter=',')
            self.perception_rewards[i], self.jerk_rewards[i] = rewards