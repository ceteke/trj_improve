from hlfd_learning.hlfd_learning.goal_model import HMMGoalModel
from hlfd_learning.hlfd_learning.data import TeachedAction
from lfd_improve.sparse2dense import QS2D
import matplotlib.pyplot as plt
import pickle, numpy as np

action = TeachedAction('/home/ceteke/Desktop/sim_demos_close')
goal_model = HMMGoalModel(action)

qs2d = QS2D(goal_model)

fail_close = pickle.load(open('/home/ceteke/Desktop/rewad_learning_data/fail_close.pk', 'rb'))
fail_rewards = []

for fail in fail_close:
    fail = action.perception_pca.transform(fail)
    fail_rewards.append(qs2d.get_reward(fail))

plt.plot(np.mean(fail_rewards, axis=0), label='fail')

success_close = pickle.load(open('/home/ceteke/Desktop/rewad_learning_data/success_close.pk', 'rb'))
success_rewards = []

for success in success_close:
    success = action.perception_pca.transform(success)
    success_rewards.append(qs2d.get_reward(success))

plt.plot(np.mean(success_rewards, axis=0), label='success')

plt.legend()
plt.show()