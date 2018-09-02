import numpy as np
from functools import lru_cache


def corr_func(data):
    """
        Calculate correlation function of data
    """
    data = np.array(data)

    data = data - data.mean()

    @lru_cache(maxsize=None)
    def func(n):
        if n < 0:
            return func(-n)
        if n == 0:
            return data.std() ** 2

        x = data[:-n] * data[n:]

        return np.sum(x) / (np.size(x) - 1)

    return func


def normalized_corr_func(data):
    """
        Calculate normalized correlation function of data
    """
    data = np.array(data)

    corr = corr_func(data)

    dispersion = corr(0)

    @lru_cache(maxsize=None)
    def func(n):
        return corr(n) / dispersion

    return func


def corr_error(corr1, corr2, n=11):
    return sum((corr1(n) - corr2(n))**2 for n in range(1, n))
