import numpy as np
from dtw import dtw
from scipy.stats import multivariate_normal
import cvxpy as cp
import scipy.optimize


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

def normal_overlap(m1, c1, m2, c2):
    sigm = (c1 + c2)/2.

    bhatt_dist = 0.125 * np.matmul(np.matmul((m1-m2).T,
                                   np.linalg.inv(sigm)),
                                   m1-m2)
    bhatt_dist += 0.5 * np.log(np.linalg.det(sigm)/np.sqrt(np.linalg.det(c1) * np.linalg.det(c2)))

    return bhatt_dist


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False