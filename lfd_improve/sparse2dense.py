import numpy as np
from numba import jit


class QS2D(object):
    def __init__(self, goal_model, n_episode=100, gamma=0.9, values=None, n_iter=25):
        self.goal_model = goal_model
        self.gamma = gamma
        self.v_table = np.zeros(self.goal_model.n_components)
        self.actions = np.arange(self.goal_model.n_components)
        self.states = np.arange(self.goal_model.n_components)

        values_iter = np.zeros((n_iter, self.goal_model.n_components))
        values_iter[:,-1] = -1.

        if values is None:
            for n in range(n_iter):
                print "{}/{}".format(n+1, n_iter)
                self.learn(values_iter[n], n_episode)

            self.v_table = np.mean(values_iter, axis=0)
            self.v_table = (self.v_table - self.v_table.min()) / (
                    self.v_table.max() - self.v_table.min())  # Normalize values

        else:
            self.v_table = values

    @jit(parallel=True)
    def learn(self, v, n_episode):
        i = 0
        while i < n_episode:
            features, states = self.goal_model.hmm.sample(self.goal_model.T)
            is_success = self.goal_model.is_success(features)

            if is_success:
                i += 1
                self.update(v, features, states)


    @jit(parallel=True)
    def update(self, v_table, features, states):
        is_success = self.goal_model.is_success(features)
        rewards = [0.] * len(states)
        rewards[-1] = 1.0 if is_success else -1.0

        for t in range(len(states)):
            s = states[t]
            r = rewards[t]

            qs = []
            for a in self.actions:
                v = 0.0
                for s_prime in self.states:
                    v += self.goal_model.hmm.transmat_[s][s_prime] * v_table[s_prime]
                qs.append(r + self.gamma * v)

            v_table[s] = np.max(qs)

    def get_reward(self, per_seq):
        states = self.goal_model.hmm.predict(per_seq)
        rewards = []

        for t in range(len(states)):
            s = states[t]
            rewards.append(self.v_table[s])

        return np.array(rewards)