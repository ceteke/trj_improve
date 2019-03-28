import numpy as np
from lfd_improve.learning import TrajectoryLearning
import matplotlib.pyplot as plt

cov = np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex144/1/1/cma_cov.csv', delimiter=',')
w = np.loadtxt('/home/ceteke/Desktop/lfd_improve_demos_sim/open/ex144/1/1/w.csv', delimiter=',')

stds = np.sqrt(np.diag(cov))
stds = stds.reshape(7, 10)[:3,:].mean(axis=0)

min_var = np.argmin(stds)

tl = TrajectoryLearning('/home/ceteke/Desktop/lfd_improve_demos_sim/open', 10, 200, 10, 5, True)
tl.dmp.w = w

min_var_basis = tl.dmp.psi[min_var]
min_var_t_idx = np.argmax(min_var_basis)

print min_var_t_idx
min_not_active_idx = np.where(min_var_basis < 1e-1)[0]
next_not_active_idx = min_not_active_idx[np.where(min_not_active_idx > min_var_t_idx)[0][0]]
print next_not_active_idx

t, x, _, _ = tl.dmp.imitate()

plt.plot(x[:min_var_t_idx,0], x[:min_var_t_idx,1], linestyle=':', color='black')
plt.plot(x[min_var_t_idx:next_not_active_idx,0], x[min_var_t_idx:next_not_active_idx,1], color='black')
plt.plot(x[next_not_active_idx:,0], x[next_not_active_idx:,1], linestyle=':', color='black')
plt.show()
