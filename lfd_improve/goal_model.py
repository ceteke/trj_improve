import warnings
from sklearn.mixture import GaussianMixture
import numpy as np
from hmmlearn.hmm import GaussianHMM
from utils import pdf_multivariate



class HMMGoalModel(object):
    def __init__(self, per_seq):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        components = [2,4,6,8,10]
        n_points, self.n_dims = per_seq.shape

        hmms = [GaussianHMM(n_components=c) for c in components]

        map(lambda g: g.fit(per_seq), hmms)
        scores = map(lambda g: g.score(per_seq), hmms)

        max_score, self.hmm = sorted(zip(scores, hmms))[-1]
        print "Goal HMM n_components", self.hmm.n_components
        self.final_states = np.array(self.hmm.predict(per_seq)).reshape(-1,n_points)[:,-1]

        self.n_components = self.hmm.n_components
        self.beliefs = np.zeros((n_points+1,  self.n_components))
        self.beliefs[0] = self.hmm.startprob_

        for i in range( self.n_components):
            for t in range(1, n_points+1): # +1 since we have 0th time too
                for j in range( self.n_components):
                    self.beliefs[t, i] += self.beliefs[t-1, j] * self.hmm.transmat_[j, i]

        self.T = n_points

    def align_state(self, seq):

        if len(seq) > self.T:
            i = 0
            while i < len(seq) and len(seq) > self.T:
                seq[i] = (seq[i] + seq[i+1]) / 2.
                seq = np.delete(seq, i+1, axis=0)
                i += 1
        elif len(seq) < self.T: #TODO: Fix
            pass
        else:
            pass

        return seq

    def get_observation_prob(self, o, t):
        assert t > 0, "t can't be 0 or negative"
        total = 0.0
        for i in range(self.n_components):
            for j in range(self.n_components):
                a_ji = self.hmm.transmat_[j, i]
                b_it = self.b_i(o, i)
                total += a_ji*b_it*self.beliefs[t-1,j]
        return total

    def seq2prob(self, seq):
        probs = []
        for t in range(1,len(seq)+1):
            probs.append(self.get_observation_prob(seq[t-1], t))
        return np.array(probs)

    def b_i(self, o_t, i):
        return pdf_multivariate(o_t, self.hmm.means_[i], self.hmm.covars_[i])

    def is_success(self, per_trj):
        per_trj = np.array(per_trj)
        if per_trj.shape[-1] != self.n_dims:
            per_trj = self.action.perception_pca.transform(per_trj)
        states = self.hmm.predict(per_trj)
        final_state = states[-1]
        return final_state in self.final_states

    def sample(self, n, pos=True, t=None):
        t = self.T if t is None else t

        samples = []

        while len(samples) < n:
            s = self.hmm.sample(t)[0]
            is_success = self.is_success(s)
            if pos:
                if is_success:
                    samples.append(s)
            else:
                if not is_success:
                    samples.append(s)

        return np.array(samples)

    def posterior(self, seq):
        return self.hmm.predict_proba(seq)