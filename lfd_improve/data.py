import pickle, os, numpy as np

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