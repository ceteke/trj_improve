import open3d as o3d
import pickle
from lfd_improve.sparse2dense import QS2D
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from lfd_improve.data import MultiDemonstration


class RenamingUnpickler(pickle.Unpickler, object):
    def find_class(self, module, name):
        if module == 'lfd_improve.lfd_improve.goal_model':
            module = 'lfd_improve.goal_model'
        return super(RenamingUnpickler, self).find_class(module, name)

demo = "/Volumes/Feyyaz/MSc/lfd_improve_demos/close/"
ex_dir = os.path.join(demo, 'ex1')

values = np.loadtxt(os.path.join(ex_dir, 'values.csv'))
values[np.argmax(values)] /= 10

with open(os.path.join(ex_dir, 'goal_model.pk'), 'rb') as fp:
    goal_model = RenamingUnpickler(fp).load()

s2d = QS2D(goal_model, values=values)

pc_data = pickle.load(open(os.path.join(demo, '1', 'pcae.pk'), 'rb'))[1:]
pc_ts = np.array([p[0] for p in pc_data])
pc_feats = np.array([p[1] for p in pc_data])
pc_ts -= pc_ts[0]

print pc_ts.shape, pc_feats.shape

data = MultiDemonstration(demo)

pca = PCA(n_components=8)

per_lens = list(map(lambda d: len(d.per_feats), data.demos))
per_feats = np.concatenate([d.per_feats for d in data.demos])

pca.fit(per_feats)
pc_feats = pca.transform(pc_feats)

plt.scatter(pc_feats[:, 0], pc_feats[:, 1])
plt.show()

returns = []
for t in range(len(pc_feats)):
    pc_seq = pc_feats[:t+1]
    returns.append(s2d.get_expected_return(pc_seq))

plt.xticks(range(0, int(max(pc_ts))+1))
plt.plot(pc_ts, returns)
plt.title("Close")
plt.xlabel('Time (s)')
plt.ylabel("Return")
plt.savefig('/Users/cem/Desktop/close_return.png', bbox_inches="tight", dpi=400)
plt.show()

ts = [0, 4, 5.25, 6, 7, pc_ts[-1]]

for t in ts:
    print np.argmin(np.abs(pc_ts-t))

#print("Load a ply point cloud, print it, and render it")
#pcd = o3d.io.read_point_cloud("/Volumes/Feyyaz/MSc/lfd_improve_demos/open2/1/pf_0_seg.pcd")
#o3d.visualization.draw_geometries([pcd])