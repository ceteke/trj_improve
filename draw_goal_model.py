from lfd_improve.learning import TrajectoryLearning
import pygraphviz as pgv
import os
from open3d import *
import pickle

class RenamingUnpickler(pickle.Unpickler, object):
    def find_class(self, module, name):
        if module == 'lfd_improve.lfd_improve.goal_model':
            module = 'lfd_improve.goal_model'
        return super(RenamingUnpickler, self).find_class(module, name)

demo_dir = '/home/ceteke/Desktop/lfd_improve_demos/open2'
val_dir = os.path.join(demo_dir, 'ex1', 'values.csv')
goal_model_dir = os.path.join(demo_dir, 'ex1', 'goal_model.pk')

vals = np.loadtxt(val_dir)

with open(goal_model_dir, 'rb') as fp:
    gm = RenamingUnpickler(fp).load()

tl1 = TrajectoryLearning(demo_dir, 10, 150, 1, 5, False, adaptive_covar=True,
                         values=vals, goal_model=gm)

goal_model = tl1.goal_model

G = pgv.AGraph(strict=False, directed=True)

T = goal_model.hmm.transmat_
visited = []
start = None

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        t = round(T[i,j], 2)
        if t > 1e-10 and (i,j) not in visited:
            G.add_edge(i, j, weight=t, label=t)
            visited.append((i,j))
    pi = round(goal_model.hmm.startprob_[i], 2)
    if pi == 1:
        start = i

for s in goal_model.final_states:
    n = G.get_node(s)
    n.attr['color'] = 'green'

n = G.get_node(start)
n.attr['color'] = 'blue'

for n in G.nodes():
    v = round(tl1.s2d.v_table[int(n)], 2)
    n.attr['label'] = n+" ({})".format(v)

# Get closest point clouds to hmm centers
per_seq = tl1.per_data[:tl1.per_lens[0]]
mean_pc_idxs = []

for k, mu in enumerate(goal_model.hmm.means_):
    closest = np.argmin(np.linalg.norm(per_seq - mu, axis=1))
    mean_pc_idxs.append(closest)

pc_paths = []
for mu_idx in mean_pc_idxs:
    fname = 'pf_{}_seg.pcd'.format(mu_idx)
    print mu_idx
    path = os.path.join(tl1.data_dir, '1', fname)
    pc_paths.append(path)

G.layout('dot')
G.draw('g.png')

while True:
    sid = int(raw_input('Enter state id: '))
    pc_path = pc_paths[sid]
    pc = read_point_cloud(pc_path)
    draw_geometries([pc])
