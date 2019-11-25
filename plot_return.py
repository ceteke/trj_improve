import pickle
from lfd_improve.sparse2dense import QS2D
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from lfd_improve.data import MultiDemonstration


class RenamingUnpickler(pickle.Unpickler, object):
    def find_class(self, module, name):
        if module == 'lfd_improve.lfd_improve.goal_model':
            module = 'lfd_improve.goal_model'
        return super(RenamingUnpickler, self).find_class(module, name)

demo = "/Users/cem/Documents/lfd_improve_demos/open/"
data = "/Users/cem/Documents/lfd_improve_demos/draw/"
ex_dir = os.path.join(data, '1')

values = np.loadtxt(os.path.join(demo, 'values.csv'))
# values[4] *= 10
values[np.argmax(values)] /= 10

with open(os.path.join(demo, 'goal_model.pk'), 'rb') as fp:
    goal_model = RenamingUnpickler(fp).load()

s2d = QS2D(goal_model, values=values)

pc_data = pickle.load(open(os.path.join(ex_dir, 'pcae.pk'), 'rb'))[1:]
pc_ts = np.array([p[0] for p in pc_data])
pc_feats = np.array([p[1] for p in pc_data])
pc_ts -= pc_ts[0]

data = MultiDemonstration(demo)

pca = PCA(n_components=8)

per_lens = list(map(lambda d: len(d.per_feats), data.demos))
per_feats = np.concatenate([d.per_feats for d in data.demos])

pca.fit(per_feats)
pc_feats = pca.transform(pc_feats)

#plt.scatter(pc_feats[:, 0], pc_feats[:, 1])
#plt.show()

returns = s2d.get_expected_return(pc_feats, arr=True)
all_returns = np.zeros((len(pc_feats), len(pc_feats))) - 1

plt.ylim(-0.1, 1.1)
plt.xticks(range(0, int(max(pc_ts))+1))
plt.plot(pc_ts, returns)
plt.show()

# for t in range(len(pc_feats)):
#     pc_seq = pc_feats[:t+1]
#
#     rew_arr = s2d.get_expected_return(pc_seq, arr=True)
#     all_returns[t, :t + 1] = rew_arr
#     returns.append(np.sum(rew_arr))

# im = plt.imshow(all_returns)
# plt.colorbar(im)
# plt.show()

# pc1 = pc_feats[:54]
# pc2 = pc_feats[:55]

# pr1 = goal_model.hmm.predict_proba(pc1).T
# pr2 = goal_model.hmm.predict_proba(pc2)[:-1].T

# rew_arr1 = s2d.get_expected_return(pc1, arr=True)
# rew_arr2 = s2d.get_expected_return(pc2, arr=True)

# f1, ax = plt.subplots(2)

# im = ax[0].imshow(pr1)
# ax[1].imshow(pr2)

# f1.subplots_adjust(right=0.8)
# cbar_ax = f1.add_axes([0.85, 0.15, 0.05, 0.7])
# f1.colorbar(im, cax=cbar_ax)

# plt.show()

# plt.title("Close")
# plt.xlabel('Time (s)')
# plt.ylabel("Return")
#plt.savefig('/Users/cem/Desktop/close_return.png', bbox_inches="tight", dpi=400)

#
# ts = [0, 4, 5.25, 6, 7, pc_ts[-1]]
#
# for t in ts:
#     print np.argmin(np.abs(pc_ts-t))

#print("Load a ply point cloud, print it, and render it")
#pcd = o3d.io.read_point_cloud("/Volumes/Feyyaz/MSc/lfd_improve_demos/open2/1/pf_0_seg.pcd")
#o3d.visualization.draw_geometries([pcd])