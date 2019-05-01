import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.manifold import TSNE


def load_feats(dir):
    per_inf = pickle.load(open(dir, 'rb'))[1:]
    ts = np.array([p[0] for p in per_inf])
    ts -= ts[0]
    return ts, np.array([p[1] for p in per_inf])

t1, p1 = load_feats('/home/ceteke/Desktop/lfd_improve_demos/0/1/pcae.pk')
t2, p2 = load_feats('/home/ceteke/Desktop/lfd_improve_demos/0/2/pcae.pk')
t3, p3 = load_feats('/home/ceteke/Desktop/lfd_improve_demos/0/3/pcae.pk')

all_dat = np.concatenate((p1,p2,p3), axis=0)
print all_dat.shape

tsne = TSNE(n_components=3)
latent = tsne.fit_transform(p1)

y = ndimage.median_filter(latent[:,0], 7)

plt.plot(y)
plt.show()