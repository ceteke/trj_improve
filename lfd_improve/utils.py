import numpy as np
from dtw import dtw


def pdf_multivariate(x, mu, cov):
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))

def get_jerk_reward(dddx):
    total_jerk = np.linalg.norm(dddx, axis=1).sum()
    return 1. / total_jerk

def align_trajectories(data, longest=True):
    data = list(map(np.array, data))
    if longest:
        ls = np.argmax([d.shape[0] for d in data])
    else:
        ls = np.argmin([d.shape[0] for d in data])

    data_warp = []

    for j, d in enumerate(data):
        dist, cost, acc, path = dtw(data[ls], d,
                                    dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        data_warp += [d[path[1]][:data[ls].shape[0]]]

    return np.array(data_warp)

def sampling_fn(r, lb, a, h):
    return lb + a * (np.exp(-np.square(np.log(r+1e-10)/h)))