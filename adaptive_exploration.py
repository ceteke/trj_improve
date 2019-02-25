from lfd_improve.utils import sampling_fn
import numpy as np
import matplotlib.pyplot as plt

rew = np.arange(0,1,0.01)
hs = [0.25, 0.5, 0.75, 1., 1.5, 1.75, 2., 2.5, 3.]

for h in hs:
    y = sampling_fn(rew, 0, 25, h)
    plt.plot(rew, y, label=str(h))

plt.legend()
plt.show()