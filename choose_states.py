from lfd_improve.data import MultiDemonstration
import numpy as np
from sklearn.decomposition import PCA
from lfd_improve.goal_model import HMMGoalModel
from lfd_improve.sparse2dense import QS2D

skill = MultiDemonstration('/home/ceteke/Desktop/lfd_improve_demos_sim/open')

per_lens = list(map(lambda d: len(d.per_feats), skill.demos))
per_feats = np.concatenate([d.per_feats for d in skill.demos])

pca = PCA(8)
per_data = pca.fit_transform(per_feats)

goal_model = HMMGoalModel(per_data, per_lens)
s2d = QS2D(goal_model)