from lfd_improve.experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


# skill_dir = '/home/ceteke/Desktop/lfd_improve_demos_sim/open'

# experiment_names_sparse = {
#     'PoWER': range(314,334),
#     'PI2-ES': range(274,294),
#     'PI2-ES-Cov': range(254,274),
# }
#
# experiment_names_dense = {
#     'PoWER': range(334,354),
#     'PI2-ES': range(294,314),
#     'PI2-ES-Cov': range(234,254),
# }

skill_dir = '/home/ceteke/Desktop/lfd_improve_demos_sim/close'

experiment_names_sparse = {
    'PoWER': range(132,152),
    'PI2-ES': range(72,92),
    'PI2-ES-Cov': range(32,52),
}

experiment_names_dense = {
    'PoWER': range(112,132),
    'PI2-ES': range(92,112),
    'PI2-ES-Cov': range(52,72),
}

markers = {
    'PoWER': '1',
    'PI2-ES': '2',
    'PI2-ES-Cov': '3'
}


# Sparse vs Dense

for method, idxs_sparse in experiment_names_sparse.items():
    idxs_dense = experiment_names_dense[method]

    experiments_sparse = [Experiment('{}/ex{}'.format(skill_dir, i)) for i in idxs_sparse]
    experiments_dense = [Experiment('{}/ex{}'.format(skill_dir, i)) for i in idxs_dense]

    success_sparse = np.array(map(lambda e: e.successes_greedy, experiments_sparse))
    success_dense = np.array(map(lambda e: e.successes_greedy, experiments_dense))

    sparse_mean = np.mean(success_sparse, axis=0)
    sparse_var = np.var(success_sparse, axis=0)

    dense_mean = np.mean(success_dense, axis=0)
    dense_var = np.var(success_dense, axis=0)

    plt.title(method)

    X = list(range(1, len(dense_mean)+1))

    plt.plot(X, sparse_mean, label='Sparse', marker=markers[method], markersize=16, linestyle=':')
    plt.plot(X, dense_mean, label='Dense', marker=markers[method], markersize=16)

    plt.fill_between(X, np.clip(sparse_mean+sparse_var, 0, 1), np.clip(sparse_mean-sparse_var, 0, 1), alpha=0.2)
    plt.fill_between(X, np.clip(dense_mean+dense_var, 0, 1), np.clip(dense_mean-dense_var,0, 1), alpha=0.2)

    plt.ylim((0, 1.01))
    plt.xlabel('Greedy')
    plt.ylabel('Success')
    plt.legend()
    plt.savefig('/home/ceteke/Desktop/{}.png'.format(method.lower()), bbox_inches="tight", dpi=400)
    plt.show()


# Compare methods (Dense)
for method, idxs_dense in experiment_names_dense.items():
    experiments_dense = [Experiment('{}/ex{}'.format(skill_dir, i)) for i in idxs_dense]
    success_dense = np.array(map(lambda e: e.successes_greedy, experiments_dense))

    dense_mean = np.mean(success_dense, axis=0)
    dense_var = np.var(success_dense, axis=0)

    X = list(range(1, len(dense_mean) + 1))
    plt.plot(X, dense_mean, label=method, marker=markers[method], markersize=16)

plt.ylim((0, 1.01))
plt.xlabel('Greedy')
plt.ylabel('Success')
plt.legend()
plt.title("Close")
plt.savefig('/home/ceteke/Desktop/means.png', bbox_inches="tight", dpi=400)
plt.show()