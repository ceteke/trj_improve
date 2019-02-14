from dmp.rl import DMPPower
from data import Demonstration
from sparse2dense import QS2D
from goal_model import HMMGoalModel
import numpy as np
from sklearn.decomposition import PCA
from spliner import Spliner


class TrajectoryLearning(object):
    def __init__(self, demo_dir, n_basis, K, n_sample, n_episode, n_perception=8, alpha=1., beta=.25):
        self.dmp = DMPPower(n_basis, K, n_sample)
        self.demo = Demonstration(demo_dir)
        self.n_episode = n_episode

        self.pca = PCA(n_components=n_perception)
        per_data = self.pca.fit_transform(self.demo.per_feats)

        self.goal_model = HMMGoalModel(per_data)
        self.s2d = QS2D(self.goal_model)

        self.delta = np.diff(self.demo.times).mean()
        print "Delta:", self.delta

        self.spliner = Spliner(self.demo.times, self.demo.ee_poses)
        t_gold, y_gold, yd_gold, ydd_gold, yddd_gold = self.spliner.get_motion

        self.dmp.fit(t_gold, y_gold, yd_gold, ydd_gold)

        self.e = 0
        self.std = self.dmp.w.std(axis=1).mean()
        self.std_initial = self.std
        self.decay_episode = self.n_episode / 2.
        self.n_perception = n_perception

        self.alpha = alpha
        self.beta = beta * (self.s2d.v_table.max()/self.get_jerk_reward(t_gold, y_gold))

        print "Beta:", self.beta

    def decay_std(self, initial):
        if self.e >= self.decay_episode:
            return initial * (1. - ((self.e - self.decay_episode) / (self.n_episode - self.decay_episode)))
        return initial

    def generate_episode(self):
        print "STD:", self.std

        self.e += 1
        episode = self.dmp.generate_episode(self.std)
        self.std = self.decay_std(self.std_initial)

        return episode

    def get_jerk_reward(self, t, x):
        episode_spliner = Spliner(t, x)
        _, _, _, _, dddx_episode = episode_spliner.get_motion
        total_jerk = np.linalg.norm(dddx_episode, axis=1).sum()
        return 1. / total_jerk

    def get_reward(self, per_trj, jerk):
        perception_reward = self.s2d.get_reward(per_trj)[-1]

        if jerk:
            jerk_reward = self.get_jerk_reward(self.dmp.t_episode, self.dmp.x_episode)
        else:
            jerk_reward = 0.0

        total_reward = self.alpha*perception_reward + self.beta*jerk_reward

        print "\tJerk Reward:", jerk_reward
        print "\tPerception Reward:", perception_reward
        print "\tTotal Reward:", total_reward

        return total_reward

    def update(self, per_trj, jerk=True):
        if per_trj is not None:
            if per_trj.shape[-1] != self.n_perception:
                per_trj = self.pca.transform(per_trj)

            reward = self.get_reward(per_trj, jerk)
        else:
            reward = 0

        self.dmp.update(reward)

        return reward