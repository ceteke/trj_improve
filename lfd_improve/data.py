import pickle, os, numpy as np
from spliner import Spliner


class Demonstration(object):
    def __init__(self, demo_dir):
        self.demo_dir = demo_dir
        robot_dir = os.path.join(self.demo_dir, 'ee_poses_wrt_obj.csv')
        per_dir = os.path.join(self.demo_dir, 'perception.csv')
        torque_dir = os.path.join(self.demo_dir, 'joint_torques.csv')

        robot_data = np.loadtxt(robot_dir, delimiter=',')
        per_data = np.loadtxt(per_dir, delimiter=',')
        torque_data = np.loadtxt(torque_dir, delimiter=',')

        self.times = per_data[:, 0]
        self.ee_poses = robot_data[:, 1:]
        self.per_feats = per_data[:, 1:]
        self.torques = torque_data[:, 1:]

        self.spliner = Spliner(self.times, self.ee_poses)
        self.t, self.x, self.dx, self.ddx, self.dddx = self.spliner.get_motion

class MultiDemonstration(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.demo_dirs = list(
            map(lambda x: os.path.join(data_dir, x),
                filter(lambda x: str.isdigit(x), os.listdir(data_dir))))
        self.demo_dirs = sorted(self.demo_dirs,
                                   key=lambda x: int(os.path.basename(os.path.normpath(x))))

        self.demos = [Demonstration(d) for d in self.demo_dirs]