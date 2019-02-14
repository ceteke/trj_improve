import pickle
import numpy as np
import matplotlib.pyplot as plt
from dmp.rl import DMPPower
import scipy.signal
from lfd_improve.spliner import Spliner

robot_data = pickle.load(open('/home/ceteke/Desktop/dmp_improve_demos/open/1/robot_states.pk', 'rb'))[1:]
times = np.array([r[0] for r in robot_data])
ee_poses = np.array([r[4] for r in robot_data])
times = times - times[0]

dmp = DMPPower(20,100,5)
spliner = Spliner(times, ee_poses)

ts, y_gold, yd_gold, ydd_gold, yddd_gold = spliner.get_motion

print np.linalg.norm(yddd_gold, axis=1).mean()
print '--'
dmp.fit(ts, y_gold, yd_gold, ydd_gold)


f,axs = plt.subplots(3)

for i in range(3):
    axs[i].plot(ts, ee_poses[:,i], linestyle=':')

ts, p_dmp, _, ddp_dmp = dmp.imitate()

for i in range(3):
    axs[i].plot(ts, p_dmp[:,i], linestyle='-.')

dmp_spliner = Spliner(ts, p_dmp)
_,_,_,_,jerks = dmp_spliner.get_motion

print np.linalg.norm(jerks, axis=1).sum()

ts, p_dmp, _, ddp_dmp = dmp.imitate(w=np.loadtxt('/home/ceteke/Desktop/ex_dmp/dmp7.csv'))

for i in range(3):
    axs[i].plot(ts, p_dmp[:,i])

dmp_improved_spliner = Spliner(ts, p_dmp)
_,_,_,_,jerks = dmp_improved_spliner.get_motion

print np.linalg.norm(jerks, axis=1).sum()
print np.loadtxt('/home/ceteke/Desktop/ex_dmp/dmp20.csv').std(axis=1)
plt.show()