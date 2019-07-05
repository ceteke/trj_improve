import pickle
import numpy as np
import matplotlib.pyplot as plt
from lfd_improve.goal_model import HMMGoalModel



torques = [
    '/home/ceteke/Desktop/lfd_improve_demos/draw/1/joint_torques.pk',
    '/home/ceteke/Desktop/lfd_improve_demos/draw/2/joint_torques.pk'
]

joint_torques = []
torque_times = []

for t in torques:
    toq = pickle.load(open(t, 'rb'))[1:]
    joint_torques.append(np.array([tf[1] for tf in toq]))
    torque_times.append(np.array([tf[0] for tf in toq]))


for i, jt in enumerate(torque_times):
    torque_times[i] -= torque_times[i][0]

f, axs = plt.subplots(3)

for i in range(len(joint_torques)):
    for k in range(3):
        axs[k].plot(torque_times[i],
                    joint_torques[i][:, k])

plt.show()

goal_model = HMMGoalModel(np.concatenate(joint_torques, axis=0), map(len, joint_torques))