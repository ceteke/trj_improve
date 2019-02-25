from lfd_improve.utils import sampling_fn
import numpy as np
import matplotlib.pyplot as plt

rew = np.arange(0,1,0.01)
hs = [0.25, 0.5, 0.75, 1., 1.5, 1.75, 2., 2.5, 3.]

y = sampling_fn(1-rew, 0, 15, .75)
plt.plot(rew, y)

plt.show()