from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt, pickle
import numpy as np


tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos/open', 15, 200, 5, 5, False)

fail_feats = np.array(
    [p[1] for p in pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/open/ex2/1/5/pcae.pk', 'rb'))])
fail_feats = tl.pca.transform(fail_feats)
print tl.goal_model.is_success(fail_feats)

print tl.s2d.v_table

for i in range(len(tl.t_gold)):
    demo = tl.x_gold[i]
    plt.plot(demo[:,0], demo[:,1], label='demo{}'.format(i+1), linestyle='-.')

for _ in range(10):
    t, x, _, _ = tl.dmp.generate_episode()
    plt.plot(x[:,0], x[:,1], linestyle=':', alpha=0.5)

plt.plot(tl.x_imitate[:,0], tl.x_imitate[:,1], label='imitate', color='black')
plt.legend()
plt.show()