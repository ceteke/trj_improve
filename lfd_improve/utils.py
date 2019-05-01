import numpy as np
from fastdtw import fastdtw


def aic(hmm, per_data, per_lens, mean=True):
    lower = 0
    n_states = hmm.n_components
    n_dim = per_data.shape[-1]
    n_params = 0.0

    # Means
    n_params += n_states * n_dim
    # Covars Diagonal
    n_params += n_states * n_dim
    # Trans mat
    n_params += n_states ** 2
    # Priors
    n_params += n_states

    total_aic = 0.0

    for i in range(len(per_lens)):
        data = per_data[lower:lower + per_lens[i]]
        lower += per_lens[i]
        ll = hmm.score(data)
        aic = 2 * (n_params - ll)
        total_aic += aic

    return total_aic / len(per_lens) if mean else total_aic

def update_gmm(gmm, centers, covars, weights):
    gmm.means_ = centers
    gmm.covariances_ = covars
    gmm.weights_ = weights
    gmm.n_components = len(centers)
    gmm.precisions_ = np.zeros_like(gmm.covariances_)
    gmm.precisions_cholesky_ = np.zeros_like(gmm.covariances_)

    for i in range(gmm.n_components):
        gmm.precisions_[i] = np.linalg.inv(gmm.covariances_[i])
        try:
            gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.precisions_[i])
        except np.linalg.LinAlgError:
            gmm.precisions_cholesky_[i] = np.linalg.cholesky(
                nearestPD(gmm.precisions_[i])
            )

def pdf_multivariate(x, mu, cov):
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))

def get_jerk_reward(dddx):
    total_jerk = np.linalg.norm(dddx, axis=1).sum()
    return 1. / total_jerk

def align_trajectories(times, ee_poses, longest=True):
    def dist(x, y):
        euc = np.linalg.norm(x[:3]-y[:3])

        x_quat = x[4:]
        y_quat = y[4:]

        if np.linalg.norm(x_quat) != 1:
            x_quat /= np.linalg.norm(x_quat)

        if np.linalg.norm(y_quat) != 1:
            y_quat /= np.linalg.norm(y_quat)

        theta = 1 - x_quat.dot(y_quat) ** 2

        return euc + theta

    met = np.argmax if longest else np.argmin
    t0_idx = met(map(len, times))
    ee0 = ee_poses[t0_idx]
    t0 = times[t0_idx]

    ee_poses_aligned = np.zeros((len(ee_poses), len(ee0), 7))

    for i in range(len(times)):
        if i != t0_idx:
            distance, path = fastdtw(ee0, ee_poses[i], dist=dist)
            align_idxs = np.array([p[1] for p in path])[:len(ee0)]
            ee_poses_aligned[i] = ee_poses[i][align_idxs]
        else:
            ee_poses_aligned[i] = ee_poses[i]

    return t0, ee_poses_aligned

def sampling_fn(r, lb, a, h):
    return lb + a * (np.exp(-np.square(np.log(r+1e-10)/h)))

def center_distance(m1, m2, a=0.5, b=0.5):
    m1_euc = m1[:3]
    m2_euc = m2[:3]
    m1_quat = m1[3:]
    m2_quat = m2[3:]

    euc_dist = np.linalg.norm(m1_euc-m2_euc)

    if np.linalg.norm(m1_quat) != 1:
        m1_quat /= np.linalg.norm(m1_quat)

    if np.linalg.norm(m2_quat) != 1:
        m2_quat /= np.linalg.norm(m2_quat)

    min_theta = 1-m1_quat.dot(m2_quat)**2

    if euc_dist < 1e-8: euc_dist = 0.0
    if min_theta < 1e-8: min_theta = 0.0

    return a*euc_dist + b*min_theta



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

def plot_dmp_episodes(dmp, axs, n=10, std=1.0):
    t, x, _, _ = dmp.imitate()

    for i in range(3):
        axs[i].plot(t, x[:,i], color='black')

    for _ in range(n):
        t, x, _, _ = dmp.generate_episode(std)
        for i in range(3):
            axs[i].plot(t, x[:, i], color='C1', linestyle=':')