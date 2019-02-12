import numpy as np

def pdf_multivariate(x, mu, cov):
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))