import pickle, os, numpy as np
from spliner import Spliner


class Demonstration(object):
    def __init__(self, demo_dir):
        self.demo_dir = demo_dir
        robot_dir = os.path.join(self.demo_dir, 'robot_states.pk')
        per_dir = os.path.join(self.demo_dir, 'pcae.pk')
        robot_data = pickle.load(open(robot_dir, 'rb'))[1:]
        per_data = pickle.load(open(per_dir, 'rb'))[1:]

        self.times = np.array([r[0] for r in robot_data])
        self.ee_poses = np.array([r[-1] for r in robot_data])
        self.per_feats = np.array([p[-1] for p in per_data])
        self.times = self.times - self.times[0]

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