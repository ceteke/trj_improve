import pickle, os
import numpy as np
from sklearn.decomposition import PCA
from lfd_improve.goal_model import HMMGoalModel
from lfd_improve.sparse2dense import QS2D
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_ellipse(pos, cov):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 4 * np.sqrt(vals)

    pts = ax.scatter(pos[0], pos[1])
    c = pts.get_facecolors()[0]

    ellip = mpl.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, lw=1, fill=False, linestyle='solid', color=c)

    ax.add_artist(ellip)


p1 = np.array([p[1] for p in pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/open/1/pcae.pk', 'rb'))[1:]])
per_ts = np.array([p[0] for p in pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/open/1/pcae.pk', 'rb'))[1:]])
p2 = np.array([p[1] for p in pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/open/2/pcae.pk', 'rb'))[1:]])
p3 = np.array([p[1] for p in pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/open/3/pcae.pk', 'rb'))[1:]])
per_ts -= per_ts[0]

p_lens = [len(p1), len(p2), len(p3)]
p_data = np.concatenate([p1,p2,p3], axis=0)
pca = PCA(8)
p_data = pca.fit_transform(p_data)

goal_model = HMMGoalModel(p_data, p_lens)
s2d = QS2D(goal_model)

pickle.dump((goal_model, s2d), open('goal_test.pk', 'wb'))

goal_model, s2d = pickle.load(open('goal_test.pk', 'rb'))

hmm = goal_model.hmm

#print goal_model.hmm.startprob_
# Plot HMM
#for s in range(goal_model.n_components):
#    plot_ellipse(goal_model.hmm.means_[s], goal_model.hmm.covars_[s])

post = hmm.predict_proba(p_data[:p_lens[0]])

rews = []
for t in range(len(post)):
    exp_rew = 0.
    for i, p_s in enumerate(post[t]):
        exp_rew += p_s * s2d.v_table[i]
    rews.append(exp_rew)
plt.plot(per_ts,rews)
plt.show()


