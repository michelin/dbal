"""
Utility functions
"""

import numpy as np
import pandas as pd

PATH = "../datasets/"


def toy_example(size=1000, dim=2, cluster=10):
    Xs = 0.5 * np.random.randn(size, dim)
    mus = 2 * np.random.random((cluster, dim))
    sigmas = 0.5 * np.random.random((cluster, dim, dim))
    for i in range(len(sigmas)):
        sigmas[i, :, :] = 0.1 * sigmas[i, :, :].transpose().dot(sigmas[i, :, :])
        sigmas[i, :, :][np.diag_indices_from(sigmas[i, :, :])] *= 10

    mults = []
    for i in range(cluster):
        mults.append(np.random.multivariate_normal(mus[i], sigmas[i, :, :], size))
    mults = np.stack(mults, -1)
    espilon = np.random.choice(cluster, size)

    Xt = mults[np.arange(size), :, espilon]

    def f(X):
        return (1 / 2) * (X[:, 0] + 0.5 * X[:, 1]) + np.exp(
            X[:, 0] * X[:, 1] - X[:, 1] ** 2 - 0.25 * X[:, 0] ** 2
        )

    return Xs, Xt, f
