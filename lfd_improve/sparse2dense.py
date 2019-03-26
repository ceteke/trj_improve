import cvxpy as cp, numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, goal_model, n_episode=100, gamma=0.9, values=None, n_iter=25, tol=1e-2):
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
                i = 0
                while i < n_episode:
                    features, states = self.goal_model.hmm.sample(self.goal_model.T)
                    is_success = self.goal_model.is_success(features)

                    if is_success:
                        i+=1
                        values_iter[n] = self.update(values_iter[n], features, states)

                if n > 1:
                    diff = np.linalg.norm(values_iter[:n-1].mean(axis=0) - values_iter[:n].mean(axis=0))
                    print diff
                    if diff < tol:
                        break

            # plt.boxplot(values_iter, labels=range(len(self.v_table)))
            # plt.show()

            self.v_table = np.mean(values_iter, axis=0)
            self.v_table = (self.v_table - self.v_table.min()) / (
                    self.v_table.max() - self.v_table.min())  # Normalize values
            # print self.v_table
            # plt.bar(list(range(len(self.v_table))), self.v_table)
            # plt.show()
        else:
            self.v_table = values

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

        return v_table

    def get_reward(self, per_seq):
        states = self.goal_model.hmm.predict(per_seq)
        rewards = []

        for t in range(len(states)):
            s = states[t]
            rewards.append(self.v_table[s])

        return np.array(rewards)