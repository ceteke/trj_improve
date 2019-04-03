from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt
import numpy as np

experiment_dir = '/home/ceteke/Desktop/lfd_improve_demos/open/ex1'

n_episode = 5
n_rollout = 5

#print np.linalg.norm(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/5/10/cma_cov.csv', delimiter=',') - np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/1/1/cma_cov.csv', delimiter=','))
#exit()
tl1 = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos_sim/open', 10, 150, None, 5, True, init_std=2)
#tl1.dmp.setC(np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex153/5/10/cma_cov.csv', delimiter=','))

t, x, _, _ = tl1.dmp.imitate()

#f, axs = plt.subplots(3)

#for i in range(3):
#    axs[i].plot(t, x[:,i], color='black', lw=2)

plt.plot(x[:,0], x[:,1], lw=2, color='black')

for _ in range(50):
    t, x, _, _ = tl1.generate_episode()
    plt.plot(x[:, 0], x[:, 1], linestyle=':')
    #for i in range(3):
    #    axs[i].plot(t, x[:,i], linestyle=':', alpha=0.5)

plt.show()