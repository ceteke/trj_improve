from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt

experiment_dir = '/home/ceteke/Desktop/lfd_improve_demos/open/ex1'

n_episode = 5
n_rollout = 5

tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos/open', 15, 200, None, 5, True, init_std=0.2)

t, x, _, _ = tl.dmp.imitate()
plt.plot(x[:,0], x[:,1], color='black', lw=2)

for _ in range(50):
    t, x, _, _ = tl.dmp.generate_episode()
    plt.plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

plt.show()