from dmp.rl import DMPPower
from data import Demonstration
from sparse2dense import QS2D
from goal_model import HMMGoalModel
import numpy as np, pickle
from sklearn.decomposition import PCA
from spliner import Spliner
import os
from utils import align_trajectories, sampling_fn

class TrajectoryLearning(object):
    def __init__(self, demo_dir, n_basis, K, n_sample, n_episode, is_sparse, n_perception=8, alpha=1., beta=.5,
                 values=None, goal_model=None, goal_data=False, succ_samples=4, h=0.75):
        '''
        :param demo_dir: Demonstration directory
        :param n_basis: Number of DMP basis
        :param K: DMP Spring constant
        :param n_sample: Number of samples for PoWER update
        :param n_episode: Number of episodes
        :param is_sparse: Do you want to use sparse rewards?
        :param n_perception: Perceptual features size
        :param alpha: Coefficient of perceptual rewards
        :param beta: Ratio between max perceptual reward and max jerk reward.
        :param values: If you don't want to learn rewards set this
        :param goal_model: If you don't want to learn a new goal model give this
        :param goal_data: If there is additional goal trajectories give them here
        :param succ_samples: Number of samples to calculate success rate for adaptive exploration rate
        :param h: Adaptive exploration rate function parameter (See adaptive_exploration.py)
        '''

        self.dmp = DMPPower(n_basis, K, n_sample)
        self.demo = Demonstration(demo_dir)
        self.pca = PCA(n_components=n_perception)

        if goal_data:
            goal_data_dir = os.path.join(demo_dir, '..', 'goal_demos')
            per_data = []

            for pk in os.listdir(goal_data_dir):
                pk = os.path.join(goal_data_dir, pk)
                per_data.append([p[1] for p in pickle.load(open(pk, 'rb'))][1:])

            per_data = align_trajectories(per_data)
            N, T, D = per_data.shape
            per_data = self.pca.fit_transform(per_data.reshape(-1, D)).reshape(N, T, -1)
        else:
            per_data = self.pca.fit_transform(self.demo.per_feats).reshape(1, len(self.demo.per_feats), -1)

        print "Perception PCA Exp. Var:", np.sum(self.pca.explained_variance_ratio_)
        self.n_episode = n_episode
        self.is_sparse = is_sparse

        if goal_model is None:
            self.goal_model = HMMGoalModel(per_data)
        else:
            self.goal_model = goal_model

        if not is_sparse:
            if values is None:
                qs2d_models = [QS2D(self.goal_model, n_episode=100) for _ in range(10)]
                qs2d_models = sorted(qs2d_models, key=lambda x: x.val_var, reverse=True)
                self.s2d = qs2d_models[0]
            else:
                self.s2d = QS2D(self.goal_model, values=values)

            print self.s2d.v_table

        self.delta = np.diff(self.demo.times).mean()
        print "Delta:", self.delta

        self.spliner = Spliner(self.demo.times, self.demo.ee_poses)
        t_gold, y_gold, yd_gold, ydd_gold, yddd_gold = self.spliner.get_motion

        self.dmp.fit(t_gold, y_gold, yd_gold, ydd_gold)

        self.e = 1.
        self.std_ub = 65.
        self.std = 40.
        self.std_initial = self.std
        self.decay_episode = float(self.n_episode // 4)
        self.n_perception = n_perception
        self.std_regular = self.std
        self.last_success = []

        self.alpha = alpha
        self.beta = beta * (1.0/self.get_jerk_reward(t_gold, y_gold))
        self.succ_samples = succ_samples
        self.h = h

    def __str__(self):
        return "N Basis: {}\nK:{}\nD:{}\nAlpha:{}\nBeta:{}\nN Sample:{}\nN Episode:{}\nSTD:{}\nDecay:{}\nIs sparse:{}".format(
            self.dmp.n_basis, self.dmp.K, self.dmp.D, self.alpha, self.beta, self.dmp.n_sample, self.n_episode, self.std_initial,
            self.decay_episode, self.is_sparse
        )

    def save_goal_model(self, dir):
        pickle.dump(self.goal_model, open(dir, 'wb'))

    def decay_std(self, initial):
        if self.e >= self.decay_episode:
            return initial * (1. - ((self.e - self.decay_episode) / (self.n_episode - self.decay_episode)))
        return initial

    def generate_episode(self):
        if self.e == 1:
            std = 0.0 # Don't explore at the first episode
        else:
            std = self.std

        print "STD:", std

        episode = self.dmp.generate_episode(std)

        return episode

    def remove_episode(self):
        self.dmp.exp_ws = self.dmp.exp_ws[:-1]

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
            perception_reward = 1.0 if is_success else 0.0
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

        self.last_success.append(per_rew)

        if len(self.last_success) > self.succ_samples:
            del self.last_success[0]  # Remove oldest

        #success_rate = float(sum(self.last_success))/len(self.last_success)
        #fail_rate = 1.0 - success_rate

        reward = per_rew + jerk_rew
        print "\tTotal Reward:", reward
        print "\tIs successful:", is_success
        #print "\tSuccess rate:", success_rate

        self.dmp.update(reward)
        self.e += 1
        exp_fac = sampling_fn(1-np.mean(self.last_success), 0., self.std_ub-self.std_initial, self.h)
        print "EXP:", exp_fac
        self.std = self.decay_std(self.std_initial) + exp_fac

        return per_rew, jerk_rew, is_success