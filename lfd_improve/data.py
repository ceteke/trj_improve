import pickle, os, numpy as np
from spliner import Spliner
import matplotlib.pyplot as plt


class Demonstration(object):
    def __init__(self, demo_dir):
        self.demo_dir = demo_dir
        robot_dir = os.path.join(self.demo_dir, 'ee_poses_wrt_obj.csv')
        per_dir = os.path.join(self.demo_dir, 'perception.csv')
        torque_dir = os.path.join(self.demo_dir, 'joint_torques.csv')

        robot_data = np.loadtxt(robot_dir, delimiter=',')
        per_data = np.loadtxt(per_dir, delimiter=',')
        torque_data = np.loadtxt(torque_dir, delimiter=',')

        n_points_torque = 150

        N = len(torque_data)
        idxs = np.arange(0, N, N // n_points_torque)
        torque_data = torque_data[idxs]

        #print("N tq: ", len(torque_data))

        times = robot_data[:, 0]
        self.ee_poses = robot_data[:, 1:]
        self.per_feats = per_data[:, 1:]
        self.torques = torque_data[:, 1:8] # Last 2 is gripper torques

        self.spliner = Spliner(times, self.ee_poses)
        self.t, self.x, self.dx, self.ddx, self.dddx = self.spliner.get_motion

        # plt.plot(self.t, self.x)
        # plt.plot(times, self.ee_poses, linestyle=':')
        # plt.show()

class MultiDemonstration(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.demo_dirs = list(
            map(lambda x: os.path.join(data_dir, x),
                filter(lambda x: str.isdigit(x), os.listdir(data_dir))))
        self.demo_dirs = sorted(self.demo_dirs,
                                   key=lambda x: int(os.path.basename(os.path.normpath(x))))

        self.demos = [Demonstration(d) for d in self.demo_dirs]