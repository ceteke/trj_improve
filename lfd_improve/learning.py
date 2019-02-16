from dmp.rl import DMPPower
from data import Demonstration
from sparse2dense import QS2D
from goal_model import HMMGoalModel
import numpy as np
from sklearn.decomposition import PCA
from spliner import Spliner


class TrajectoryLearning(object):
    def __init__(self, demo_dir, n_basis, K, n_sample, n_episode, is_sparse, n_perception=8, alpha=1., beta=.5,
                 values=None):

        self.dmp = DMPPower(n_basis, K, n_sample)
        self.demo = Demonstration(demo_dir)
        self.n_episode = n_episode
        self.is_sparse = is_sparse

        self.pca = PCA(n_components=n_perception)
        per_data = self.pca.fit_transform(self.demo.per_feats)

        self.goal_model = HMMGoalModel(per_data)

        if values is None:
            qs2d_models = [QS2D(self.goal_model) for _ in range(10)]
            qs2d_models = sorted(qs2d_models, key=lambda x: x.val_var)
            qs2d_models = sorted(qs2d_models, key=lambda x: x.reward_diff, reverse=True)
            self.s2d = qs2d_models[0]
        else:
            self.s2d = QS2D(self.goal_model, values=values)

        self.delta = np.diff(self.demo.times).mean()
        print "Delta:", self.delta

        self.spliner = Spliner(self.demo.times, self.demo.ee_poses)
        t_gold, y_gold, yd_gold, ydd_gold, yddd_gold = self.spliner.get_motion

        self.dmp.fit(t_gold, y_gold, yd_gold, ydd_gold)

        self.e = 0
        self.std = self.dmp.w.std(axis=1).mean()
        self.std_initial = self.std
        self.decay_episode = float(self.n_episode // 4)
        self.n_perception = n_perception

        self.alpha = alpha
        self.beta = beta * (self.s2d.v_table.max()/self.get_jerk_reward(t_gold, y_gold))

    def __str__(self):
        return "N Basis: {}\nK:{}\nD:{}\nAlpha:{}\nBeta:{}\nN Sample:{}\nN Episode:{}\nSTD:{}\nDecay:{}\nIs sparse:{}".format(
            self.dmp.n_basis, self.dmp.K, self.dmp.D, self.alpha, self.beta, self.dmp.n_sample, self.n_episode, self.std_initial,
            self.decay_episode, self.is_sparse
        )

    def decay_std(self, initial):
        if self.e >= self.decay_episode:
            return initial * (1. - ((self.e - self.decay_episode) / (self.n_episode - self.decay_episode)))
        return initial

    def update_std(self, episode):
        for e in range(episode):
            self.e += 1
            self.std = self.decay_std(self.std_initial)

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
        if per_trj.shape[-1] != self.n_perception:
            per_trj = self.pca.transform(per_trj)

        is_success = self.goal_model.is_success(per_trj)

        if self.is_sparse:
            perception_reward = 1.0 if is_success else -1.0
        else:
            perception_reward = self.s2d.get_reward(per_trj)[-1]

        if jerk:
            jerk_reward = self.get_jerk_reward(self.dmp.t_episode, self.dmp.x_episode)
        else:
            jerk_reward = 0.0

        perception_reward = self.alpha*perception_reward
        jerk_reward = self.beta*jerk_reward

        print "\tJerk Reward:", jerk_reward
        print "\tPerception Reward:", perception_reward

        return perception_reward, jerk_reward, is_success

    def update(self, per_trj, jerk=True):
        if per_trj is not None:
            if per_trj.shape[-1] != self.n_perception:
                per_trj = self.pca.transform(per_trj)

            per_rew, jerk_rew, is_success = self.get_reward(per_trj, jerk)

        else:
            per_rew, jerk_rew = 0., 0.
            is_success = False

        reward = per_rew + jerk_rew
        print "\tTotal Reward:", reward
        print "\tIs successful:", is_success

        self.dmp.update(reward)

        return per_rew, jerk_rew, is_success