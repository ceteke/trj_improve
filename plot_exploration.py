import numpy as np
import os
from lfd_improve.learning import TrajectoryLearning
from lfd_improve.utils import plot_dmp_episodes
import matplotlib.pyplot as plt

data_dir = '/home/ceteke/Desktop/lfd_improve_demos/open'
before_cov = np.load(os.path.join(data_dir, 'ex3/1/1/cma_cov.npy'))
after_cov = np.load(os.path.join(data_dir, 'ex3/5/6/cma_cov.npy'))
after_w = np.loadtxt(os.path.join(data_dir, 'ex3/5/6/w.csv'), delimiter=',')


model_before = TrajectoryLearning(data_dir, 10, 200, 6, 5, True)
model_after = TrajectoryLearning(data_dir, 10, 200, 6, 5, True)
model_after.dmp.exp_covars = after_cov
model_after.dmp.w = after_w

f, axs = plt.subplots(3)

plot_dmp_episodes(model_before.dmp, axs)
f.show()

f, axs = plt.subplots(3)

plot_dmp_episodes(model_after.dmp, axs)
f.show()

plt.show()