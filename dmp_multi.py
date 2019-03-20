from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt


data_dir = '/home/ceteke/Desktop/sim_demos'

tl = TrajectoryLearning(data_dir, 15, 150, 5, 5, False)

t, trj, _, _ = tl.dmp.imitate()
plt.plot(trj[:,0], trj[:,1])

for _ in range(25):
    t_g, trj_g, _, _ = tl.dmp.generate_episode()
    plt.plot(trj_g[:,0], trj_g[:,1], c='C2', linestyle=":", alpha=0.75)

plt.show()