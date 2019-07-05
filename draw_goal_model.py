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

demo_dir = '/home/ceteke/Desktop/lfd_improve_demos/draw'
val_dir = os.path.join(demo_dir, 'ex2', 'values.csv')
goal_model_dir = os.path.join(demo_dir, 'ex2', 'goal_model.pk')

vals = np.loadtxt(val_dir)

with open(goal_model_dir, 'rb') as fp:
    gm = RenamingUnpickler(fp).load()

tl1 = TrajectoryLearning(demo_dir, 10, 150, 1, 5, False,
                         values=vals, goal_model=gm)

goal_model = tl1.goal_model

# scc = pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/draw/ex1/greedy_5/pcae.pk', 'rb'))[1:]
# s_feats = np.array([s[1] for s in scc])
# #
# # fl = pickle.load(open('/home/ceteke/Desktop/lfd_improve_demos/draw/ex2/1/1/pcae.pk', 'rb'))[1:]
# # f_feats = np.array([s[1] for s in fl])
# #
# s_feats = tl1.pca.transform(s_feats)
# # f_feats = tl1.pca.transform(f_feats)
# print(tl1.goal_model.hmm.predict(s_feats))
# exit()

G = pgv.AGraph(strict=False, directed=True, dpi=300)

T = goal_model.hmm.transmat_
visited = []
start = None

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        t = round(T[i,j], 2)
        if t > 1e-10 and (i,j) not in visited:
            print(t*10)
            s = np.clip(7*t, 0, 1.5)
            G.add_edge(i, j, arrowsize=s)# label=t)
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
    if v == 10:
        v = 1.0
    n.attr['label'] = n+"\n({})".format(v)

# Get closest point clouds to hmm centers
per_seq = tl1.per_data[:tl1.per_lens[0]]
mean_pc_idxs = []

for k, mu in enumerate(goal_model.hmm.means_):
    closest = np.argmin(np.linalg.norm(per_seq - mu, axis=1))
    print(k, closest)
    mean_pc_idxs.append(closest)

pc_paths = []
for mu_idx in mean_pc_idxs:
    fname = 'pf_{}_seg.pcd'.format(mu_idx)
    print mu_idx
    path = os.path.join(tl1.data_dir, '1', fname)
    pc_paths.append(path)

G.layout('circo')
G.draw('g.png')

while True:
    sid = int(raw_input('Enter state id: '))
    pc_path = pc_paths[sid]
    pc = read_point_cloud(pc_path)
    draw_geometries([pc])
