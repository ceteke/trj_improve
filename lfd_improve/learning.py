from dmp.rl import DMPPower, DMPES, DMPPI2
from dmp.imitation import ImitationDMP
from data import MultiDemonstration
from sparse2dense import QS2D
from goal_model import HMMGoalModel
import numpy as np, pickle
from sklearn.decomposition import PCA
from spliner import Spliner
import os


class TrajectoryLearning(object):
    def __init__(self, data_dir, n_basis, K, n_sample, n_episode, is_sparse, n_perception=8, alpha=1., beta=0.,
                 values=None, goal_model=None, type='es-cov', init_std=None, n_goal_states=None):
        '''
        :param data_dir:
        :param n_basis:
        :param K:
        :param n_sample:
        :param n_episode:
        :param is_sparse:
        :param n_perception:
        :param alpha:
        :param beta:
        :param values:
        :param goal_model:
        :param type: one of power, es, es-cov or pi2
        :param init_std:
        :param n_goal_states:
        '''

        self.data_dir = data_dir
        self.type = type
        self.demo = MultiDemonstration(data_dir)
        self.n_demo = len(self.demo.demos)
        self.pca = PCA(n_components=n_perception)
        self.n_basis = n_basis

        self.per_lens = list(map(lambda d: len(d.per_feats), self.demo.demos))
        self.per_feats = np.concatenate([d.per_feats for d in self.demo.demos])

        self.per_data = self.pca.fit_transform(self.per_feats)

        print "Perception PCA Exp. Var:", np.sum(self.pca.explained_variance_ratio_)
        self.n_episode = n_episode
        self.is_sparse = is_sparse

        if goal_model is None:
            self.goal_model = HMMGoalModel(self.per_data, self.per_lens, n_goal_states)
        else:
            self.goal_model = goal_model

        print "Learning reward function..."
        if not is_sparse:
            if values is None:
                self.s2d = QS2D(self.goal_model)
            else:
                self.s2d = QS2D(self.goal_model, values=values)

            print self.s2d.v_table

        dynamics_gold = ([], [], [], [], [])

        print "Forming demonstration data..."
        for d in self.demo.demos:
            spliner = Spliner(d.times, d.ee_poses)
            dynamics_demo = spliner.get_motion
            for i, dyn in enumerate(dynamics_demo):
                dynamics_gold[i].append(dyn)

        t_gold, x_gold, dx_gold, ddx_gold, dddx_gold = dynamics_gold
        self.t_gold = t_gold
        self.x_gold = x_gold

        if init_std:
            self.std = init_std
        else:
            self.std = 1.0

        weights = np.zeros((len(t_gold), 7, self.n_basis))

        for d in range(len(t_gold)):
            dmp_single = ImitationDMP(self.n_basis, K)
            dmp_single.fit(t_gold[d], x_gold[d], dx_gold[d], ddx_gold[d])
            weights[d] = dmp_single.w

        init_cov = np.zeros((self.n_basis, 7, 7))
        for b in range(self.n_basis):
            cov_b = np.cov(weights[:,:,b], rowvar=False)
            init_cov[b] = cov_b

        self.n_sample = n_sample

        if type == 'power':
            self.dmp = DMPPower(n_basis, K, n_sample, std_init=self.std, init_cov=init_cov)
            self.minimize = False
        elif type == 'es':
            self.dmp = DMPES(n_basis, K, n_sample, std_init=self.std, init_cov=init_cov, update_cov=False)
            self.minimize = True
        elif type == 'es-cov':
            self.dmp = DMPES(n_basis, K, n_sample, std_init=self.std, init_cov=init_cov, update_cov=True)
            self.minimize = True
        elif type == 'pi2':
            self.dmp = DMPPI2(n_basis, K, n_sample, std_init=self.std, init_cov=init_cov, update_cov=False)
            self.minimize = True
        elif type == 'pi2-cov':
            self.dmp = DMPPI2(n_basis, K, n_sample, std_init=self.std, init_cov=init_cov, update_cov=True)
            self.minimize = True
        else:
            raise ValueError("Unknown Policy Search Type")


        rand_demo = 1
        print "Picked demo: ", rand_demo
        t_fit, x_fit, dx_fit, ddx_fit = t_gold[rand_demo], x_gold[rand_demo], dx_gold[rand_demo], ddx_gold[rand_demo]

        print(len(t_fit))

        self.dmp.fit(t_fit, x_fit, dx_fit, ddx_fit)

        t_imitate, x_imitate, _, _ = self.dmp.imitate()

        self.e = 1.
        self.std_initial = self.std
        self.decay_episode = float(self.n_episode // 4)
        self.n_perception = n_perception

        self.alpha = alpha
        self.beta = beta * (1.0/self.get_jerk_reward(t_imitate, x_imitate))
        self.n_basis = n_basis

    def __str__(self):

        info = "N Basis:{}\nK:{}\nD:{}\nAlpha:{}\nBeta:{}\nN Sample:{}\nN Episode:{}\nSTD:{}\nDecay:{}\nIs sparse:{}\nType:{}\nN Demo:{}".format(
                self.n_basis, self.dmp.K, self.dmp.D, self.alpha, self.beta, self.dmp.n_sample, self.n_episode, self.std_initial,
                self.decay_episode, self.is_sparse, self.type, self.n_demo
            )

        return info

    def load_params(self, log_dir):
        self.dmp.w = np.loadtxt(os.path.join(log_dir, 'w.csv'), delimiter=',')
        self.dmp.exp_covars = np.load(os.path.join(log_dir, 'cma_cov.npy'))
        self.std = np.loadtxt(os.path.join(log_dir, 'log.csv'))[-1]

    def save_goal_model(self, dir):
        pickle.dump(self.goal_model, open(dir, 'wb'))

    def decay_std(self, initial):
        if self.e >= self.decay_episode:
            return initial * (1. - ((self.e - self.decay_episode) / (self.n_episode - self.decay_episode)))
        return initial

    def generate_episode(self):
        if 'cov' in self.type:
            std = 1.
        else:
            std = self.std

        print "STD", std
        episode = self.dmp.generate_episode(std)
        return episode

    def remove_episode(self):
        self.dmp.remove_episode()

    def get_jerk_reward(self, t, x):
        episode_spliner = Spliner(t, x)
        _, _, _, _, dddx_episode = episode_spliner.get_motion
        total_jerk = np.linalg.norm(dddx_episode, axis=1).sum()
        return 1. / total_jerk

    def get_reward(self, per_trj, jerk, ts, x):
        if per_trj.shape[-1] != self.n_perception:
            per_trj = self.pca.transform(per_trj)

        is_success = self.goal_model.is_success(per_trj)
        arr = True if 'pi2' in self.type else False

        if self.is_sparse:
            if arr:
                perception_reward = np.zeros(len(per_trj))
                perception_reward[-1] = 1.0 if is_success else 0.0
            else:
                perception_reward = 1.0 if is_success else 0.0

        else:
            perception_reward = self.s2d.get_expected_return(per_trj, arr)

        if jerk:
            jerk_reward = self.get_jerk_reward(ts, x)
        else:
            jerk_reward = 0.0

        perception_reward = self.alpha*perception_reward
        jerk_reward = self.beta*jerk_reward

        print "\tJerk Reward:", jerk_reward
        print "\tPerception Reward:", np.sum(perception_reward)
        reward = perception_reward + jerk_reward
        print "\tTotal Reward:", np.sum(reward)
        print "\tIs successful:", is_success

        return perception_reward, jerk_reward, is_success

    def update(self, per_trj, ts, x, jerk=True):
        if per_trj is not None:
            if per_trj.shape[-1] != self.n_perception:
                per_trj = self.pca.transform(per_trj)

            per_rew, jerk_rew, is_success = self.get_reward(per_trj, jerk, ts, x)

        else:
            per_rew, jerk_rew = 0., 0.
            is_success = False

        reward = per_rew + jerk_rew

        status = self.dmp.update(-reward if self.minimize else reward)

        if status:
            self.e += 1
            self.std = self.decay_std(self.std_initial)

        return np.sum(reward), jerk_rew, is_success