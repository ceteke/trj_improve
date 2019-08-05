from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt
import numpy as np
import pickle

n_episode = 5
n_rollout = 5

#print np.linalg.norm(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/5/10/cma_cov.csv', delimiter=',') - np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/1/1/cma_cov.csv', delimiter=','))
#exit()
tl1 = TrajectoryLearning('/home/alive/Desktop/torque_demos/close', 11, 100, 1, 5, True, init_std=1, torque_goal=True)

#scc = pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/draw/ex2/greedy_5/pcae.pk', 'rb'))[1:]
#s_feats = np.array([s[1] for s in scc])
#
# fl = pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/draw/ex2/1/1/pcae.pk', 'rb'))[1:]
# f_feats = np.array([s[1] for s in fl])
#
#s_feats = tl1.pca.transform(s_feats)
# f_feats = tl1.pca.transform(f_feats)
#print(tl1.goal_model.hmm.predict(s_feats))
#exit()
# plt.scatter(s_feats[:, 0], s_feats[:, 1])
# plt.scatter(f_feats[:, 0], f_feats[:, 1])
#
# print(tl1.per_data.shape)
# plt.scatter(tl1.per_data[:,0], tl1.per_data[:,1])
# plt.show()
# exit()

#tl1.dmp.setC(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos/close/ex2/5/10/cma_cov.csv', delimiter=','))
#tl1.dmp.exp_covars = np.load('/home/ceteke/Desktop/lfd_improve_demos/close/ex3/5/6/cma_cov.npy')
t, x, _, _ = tl1.dmp.imitate()

coords = (0,1,2)
min_coords = min(coords)

f, axs = plt.subplots(len(coords))

for i in coords:
    axs[i-min_coords].plot(t, x[:,i], color='black', lw=2)

#plt.plot(x[:,0], x[:,1], lw=2, color='black')

for i in range(len(tl1.t_gold)):
    for j in range(3):
        axs[j].plot(tl1.t_gold[i], tl1.x_gold[i][:, j])

for _ in range(25):
    t, x, _, _ = tl1.generate_episode()
    for i in coords:
        axs[i-min_coords].plot(t, x[:,i], linestyle=':')

plt.suptitle("Sampled Trajectories (Last episode, n=25)")
plt.show()