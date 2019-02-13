from dmp.rl import DMPPower
from data import Demonstration
from sparse2dense import QS2D
from goal_model import HMMGoalModel
import numpy as np
from sklearn.decomposition import PCA
import scipy.signal
from spliner import Spliner


class TrajectoryLearning(object):
    def __init__(self, demo_dir, n_basis, K, n_sample, n_perception=8, alpha=1., beta=0.1):
        self.dmp = DMPPower(n_basis, K, n_sample)
        self.demo = Demonstration(demo_dir)

        self.pca = PCA(n_components=n_perception)
        per_data = self.pca.fit_transform(self.demo.per_feats)

        self.goal_model = HMMGoalModel(per_data)
        self.s2d = QS2D(self.goal_model)

        self.delta = np.diff(self.demo.times).mean()
        print "Delta:", self.delta

        self.spliner = Spliner(self.demo.times, self.demo.ee_poses)
        _, y_gold, yd_gold, ydd_gold, _ = self.spliner.get_motion

        y_gold = y_gold.reshape(1,-1,7)
        yd_gold = yd_gold.reshape(1,-1,7)
        ydd_gold = ydd_gold.reshape(1,-1,7)

        self.dmp.fit(self.demo.times, y_gold, yd_gold, ydd_gold)

        self.e = 0
        self.std = 20
        self.std_initial = self.std
        self.decay_rate = 0.9
        self.decay_steps = 2.
        self.n_perception = n_perception

        self.alpha = alpha
        self.beta = beta

    def sigm(self, x):
        return 1./(1.+np.exp(-x))

    def decay_std(self, initial):
        return initial * self.decay_rate ** (self.e / self.decay_steps)

    def generate_episode(self):
        print "STD:", self.std

        self.e += 1
        episode = self.dmp.generate_episode(self.std)
        self.std = self.decay_std(self.std_initial)

        return episode

    def get_reward(self, per_trj, jerk):
        if per_trj.shape[-1] != self.n_perception:
            per_trj = self.pca.transform(per_trj)

        perception_reward = self.s2d.get_reward(per_trj)[-1]
        jerk_reward = 0.0

        perception_reward = perception_reward[-1]
        total_reward = self.alpha*perception_reward + self.beta*jerk_reward

        print "\tJerk Reward:", jerk_reward
        print "\tPerception Reward:", perception_reward
        print "\tTotal Reward:", total_reward

        return total_reward

    def update(self, per_trj, jerk=False):
        if per_trj is not None:
            reward = self.get_reward(per_trj, jerk)
        else:
            reward = 0

        self.dmp.update(reward)

        return reward