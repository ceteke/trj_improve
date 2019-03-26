from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt

experiment_dir = '/home/ceteke/Desktop/lfd_improve_demos/open/ex1'

n_episode = 5
n_rollout = 5

tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos_sim/open', 10, 200, None, 5, True)

for _ in range(10):
    t, x, _, _ = tl.dmp.generate_episode()
    plt.plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

plt.show()