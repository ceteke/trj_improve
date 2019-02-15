import cvxpy as cp, numpy as np
import copy


class Sparse2Dense(object):
    def __init__(self, goal_model, C=0.1, n_sample=25):
        self.goal_model = goal_model
        self.C = C

        positives = goal_model.sample(n_sample)
        negatives = goal_model.sample(n_sample, pos=False)

        self.w = cp.Variable(positives.shape[-1])

        vals = []

        for p in positives:
            for s_p in p:
                vals.append(cp.matmul(self.w, s_p))

        for n in negatives:
            for s_n in n:
                vals.append(-(cp.matmul(self.w, s_n)))

        objective = cp.Maximize(cp.sum(vals))
        constraints = [cp.norm(self.w) <= C]

        self.prob = cp.Problem(objective, constraints)
        self.prob.solve()
        self.w = self.w.value

    def get_reward(self, states):
        return np.matmul(self.w, states.T).T


class QS2D(object):
    def __init__(self, goal_model, n_episode=100, gamma=0.9, values=None):
        self.goal_model = goal_model
        self.gamma = gamma
        self.v_table = np.zeros(self.goal_model.n_components)
        self.actions = np.arange(self.goal_model.n_components)
        self.states = np.arange(self.goal_model.n_components)

        if values is None:
            n_pos, n_neg = 0, 0

            while True:
                features, states = self.goal_model.hmm.sample(self.goal_model.T)
                is_success = self.goal_model.is_success(features)

                if is_success and n_pos == n_episode//2:
                    continue

                if is_success:
                    n_pos += 1
                else:
                    n_neg += 1

                self.update(features, states)

                if n_pos + n_neg == n_episode:
                    break

            self.v_table = (self.v_table - self.v_table.min()) / (self.v_table.max() - self.v_table.min()) # Normalize values
        else:
            self.v_table = values

        pos_sample = self.goal_model.sample(1)[0]
        neg_sample = self.goal_model.sample(1, False)[0]

        pos_reward = self.get_reward(pos_sample)
        neg_reward = self.get_reward(neg_sample)

        self.reward_diff = pos_reward.sum() - neg_reward.sum()
        self.val_var = self.v_table.var()

    def update(self, features, states):
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
                    v += self.goal_model.hmm.transmat_[s][s_prime] * self.v_table[s_prime]
                qs.append(r + self.gamma * v)

            self.v_table[s] = np.max(qs)

    def get_reward(self, per_seq):
        states = self.goal_model.hmm.predict(per_seq)
        rewards = []

        for t in range(len(states)):
            s = states[t]
            rewards.append(self.v_table[s])

        return np.array(rewards)