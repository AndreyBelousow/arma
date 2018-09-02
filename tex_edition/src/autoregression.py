import numpy as np
from correlation import *


def solve(a, b):
    """
        Solves A * x = b
        Raises an exception if no solution was found.
    """
    a = np.matrix(a)
    b = np.array(b)

    b = np.reshape(b, (-1, 1))

    x = np.linalg.solve(a, b)
    x = np.array(x)

    return x.flatten()


def make_walker_matrix(corr, n):
    """
        Returns matrix for Yule-Walker equations.
        A in: A * x = b
    """
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            R[i, j] = corr(abs(i - j))

    return np.matrix(R)


def make_walker_vector(corr, n):
    """
        Returns vector for Yule-Walker equations.
        b in: A * x = b
    """
    R = np.zeros((n, 1))

    for i in range(n):
        R[i, 0] = corr(i + 1)

    return R


def calc_model_ar(n, data):
    data = np.array(data)

    corr = corr_func(data)

    if n == 0:
        return np.sqrt([corr(0)])

    A = make_walker_matrix(corr, n)
    b = make_walker_vector(corr, n)

    betas = solve(A, b)

    tmp = corr(0) - betas.dot(b)

    if tmp < 0:
        raise ArithmeticError('No solution for ar({})'.format(n))

    alpha = np.sqrt(corr(0) - betas.dot(b))

    if n == 1:
        if abs(betas[0]) >= 1:
            raise ArithmeticError('Model is unstable ar(1)')

    if n == 2:
        if abs(betas[1]) >= 1 \
                or abs(betas[0]) >= 1 - betas[1]:
            raise ArithmeticError('Model is unstable ar(2)')

    if n == 3:
        if abs(betas[2]) >= 1 \
                or abs(betas[0] + betas[2]) >= 1 - betas[1] \
                or abs(betas[1] + betas[0] * betas[2]) >= abs(1 - betas[2]**2):
            raise ArithmeticError('Model is unstable ar(3)')

    return np.append(alpha, betas)


def theory_ncf_ar(n, data):
    coeffs = calc_model_ar(n, data)
    R = corr_func(data)

    @lru_cache(maxsize=None)
    def func(m):
        if abs(m) <= n:
            return R(m)

        if n == 0:
            return 0

        rs = np.array([func(m - x) for x in range(1, n + 1)])

        return rs.dot(coeffs[1:])

    def res(m):
        return func(m) / R(0)

    return res


def model_ar(n, data):
    coeffs = calc_model_ar(n, data)

    alpha = coeffs[0]

    if n > 0:
        betas = coeffs[1:]

    def get_model(xs, x0):
        if len(xs) != n:
            raise ValueError('Length of `xs` must be equal to {}'.format(n))

        w = np.random.normal()

        if n > 0:
            xs = np.array(xs)
            x = x0

        while True:
            yield x

            if n > 0:
                x = xs.dot(betas) + w * alpha

                xs[1:] = xs[:-1]
                xs[0] = x
            else:
                x = w * alpha

            w = np.random.normal()

    return get_model
