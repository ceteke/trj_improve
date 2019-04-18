from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt
import numpy as np

n_episode = 5
n_rollout = 5

#print np.linalg.norm(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/5/10/cma_cov.csv', delimiter=',') - np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/1/1/cma_cov.csv', delimiter=','))
#exit()
tl1 = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos_sim/open', 10, 150, 1, 5, True, adaptive_covar=True)
#tl1.dmp.setC(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/5/10/cma_cov.csv', delimiter=','))

t, x, _, _ = tl1.dmp.imitate()

coords = (0,1,2)
min_coords = min(coords)

f, axs = plt.subplots(len(coords))

for i in coords:
    axs[i-min_coords].plot(t, x[:,i], color='black', lw=2)

#plt.plot(x[:,0], x[:,1], lw=2, color='black')

for _ in range(6):
    t, x, _, _ = tl1.generate_episode()
    for i in coords:
        axs[i-min_coords].plot(t, x[:,i], linestyle=':')

plt.show()