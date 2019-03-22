from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt


tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos/open', 15, 200, 5, 5, False)

for i in range(len(tl.t_gold)):
    demo = tl.x_gold[i]
    plt.plot(demo[:,0], demo[:,1], label='demo{}'.format(i+1), linestyle='-.')

for _ in range(10):
    t, x, _, _ = tl.dmp.generate_episode()
    plt.plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

plt.plot(tl.x_imitate[:,0], tl.x_imitate[:,1], label='imitate', color='black')
plt.legend()
plt.show()