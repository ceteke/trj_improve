import pickle
import numpy as np
import matplotlib.pyplot as plt
from lfd_improve.goal_model import HMMGoalModel
from lfd_improve.sparse2dense import QS2D


torques = [
    '/home/alive/Desktop/torque_demos/close2/1/joint_torques.csv',
    '/home/alive/Desktop/torque_demos/close2/2/joint_torques.csv'
]

pers = [
    '/home/alive/Desktop/torque_demos/close2/1/perception.csv',
    '/home/alive/Desktop/torque_demos/close2/2/perception.csv'
]

joint_torques = []
torque_times = []
per_times = []

#for t in torques:
    #toq = pickle.load(open(t, 'rb'))[1:]
    #joint_torques.append(np.array([tf[1] for tf in toq]))
    #torque_times.append(np.array([tf[0] for tf in toq]))

for t in torques:
    toq = np.loadtxt(t, delimiter=',')
    joint_torques.append(toq[:, 1:8])
    torque_times.append(toq[:, 0])

for p in pers:
    toq = np.loadtxt(p, delimiter=',')
    per_times.append(toq[:, 0])

n_points = 250

for i in range(len(joint_torques)):
    N = len(joint_torques[i])
    print(N, N//n_points)
    idxs = np.arange(0, N, N//n_points)
    joint_torques[i] = joint_torques[i][idxs]
    torque_times[i] = torque_times[i][idxs]

print(list(map(len, torque_times)))

f, axs = plt.subplots(7)

colors = []
for i in range(len(joint_torques)):
    for k in range(7):
        p = axs[k].plot(torque_times[i], joint_torques[i][:, k])
    colors.append(p[0].get_color())

print(colors)

torque_data = np.concatenate(joint_torques, axis=0)
torque_lens = list(map(len, joint_torques))

torque_goal = HMMGoalModel(torque_data, torque_lens, 12)

for i in range(1):
    goal_states = torque_goal.hmm.predict(joint_torques[i])
    print(goal_states)
    prev = goal_states[0]
    cps = []

    for j in range(1, len(goal_states)):
        if prev != goal_states[j]:
            cps.append(j)
            prev = goal_states[j]

    for c in cps:
        print(np.argmin(np.abs(per_times[i] - torque_times[i][c])))
        for k in range(len(axs)):
            axs[k].axvline(x=torque_times[i][c], color=colors[i])

plt.show()

s2d = QS2D(torque_goal, n_iter=5)

for i in range(len(joint_torques)):
    returns = s2d.get_expected_return(joint_torques[i], arr=True)
    returns_t = np.array([np.sum(returns[:t]) for t in range(1, len(returns))])
    plt.plot(torque_times[i][:-1], returns_t)

plt.show()