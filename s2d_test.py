from lfd_improve.goal_model import HMMGoalModel
from hlfd_learning.hlfd_learning.data import TeachedAction
from lfd_improve.sparse2dense import QS2D
import matplotlib.pyplot as plt
import pickle, numpy as np
from sklearn.decomposition import PCA

train_per = [p[1] for p in pickle.load(open('/home/ceteke/Desktop/sim_demos_close/1/pcae.pk', 'rb'))]
pca = PCA(n_components=8)
train_per = pca.fit_transform(train_per)

goal_model = HMMGoalModel(train_per.reshape(1,-1,8))

qs2d_models = [QS2D(goal_model, n_episode=100) for _ in range(10)]
qs2d_models = sorted(qs2d_models, key=lambda x: x.val_var, reverse=True)

qs2d = qs2d_models[0]
print qs2d.v_table
succ_idx = np.argmax(qs2d.v_table)
print goal_model.hmm.transmat_[:,succ_idx]

fail_close = pickle.load(open('/home/ceteke/Desktop/rewad_learning_data/fail_close.pk', 'rb'))
fail_rewards = []

for fail in fail_close:
    fail = pca.transform(fail)
    fail_rewards.append(qs2d.get_reward(fail))

plt.plot(np.mean(fail_rewards, axis=0), label='fail')

success_close = pickle.load(open('/home/ceteke/Desktop/rewad_learning_data/success_close.pk', 'rb'))
success_rewards = []

for success in success_close:
    success = pca.transform(success)
    success_rewards.append(qs2d.get_reward(success))

plt.plot(np.mean(success_rewards, axis=0), label='success')

plt.legend()
plt.show()