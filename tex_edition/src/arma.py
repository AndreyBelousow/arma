from correlation import *
from autoregression import *
from moving_average import *


def calc_model_arma(m, n, data, eps=1e-5, maxiter=100):

    data = np.array(data)

    matrix = np.zeros((m, m))
    vector = np.zeros((m, 1))

    R = corr_func(data)

    for i in range(m):
        vector[i] = R(n + i + 1)
        for j in range(m):
            matrix[i, j] = R(n + i - j)

    betas = solve(matrix, vector)

    def gen_seq():
        for i in range(m + 1, len(data)):
            xs = data[i - m - 1:i]
            y = xs[0] - xs[1:].dot(betas)

            yield y

    seq = gen_seq()
    seq_data = [x for x in seq]

    cf_seq = corr_func(seq_data)

    print('Start cf for temp. seq.')
    for i in range(11):
        print(i, cf_seq(i))
    print('End cf for temp. seq.')

    alphas = calc_model_ma(n, seq_data, eps, maxiter)

    if m == 1:
        if abs(betas[0]) >= 1:
            raise ArithmeticError('Solution unstable, betas: {}'.format(betas))
    if m == 2:
        if abs(betas[1]) >= 1 \
                or abs(betas[0]) >= 1 - betas[1]:
            raise ArithmeticError('Solution unstable, betas: {}'.format(betas))
    if m == 3:
        if abs(betas[2]) >= 1 \
                or abs(betas[0] + betas[2]) >= 1 - betas[1] \
                or abs(betas[1] + betas[0] * betas[2]) >= 1 - betas[2]**2:
            raise ArithmeticError('Solution unstable, betas: {}'.format(betas))

    return betas, alphas


def theory_ncf_arma(m, n, data, eps=1e-5, maxiter=100):

    from functools import lru_cache

    coeffs = calc_model_arma(m, n, data, eps, maxiter)

    ar_coeffs, ma_coeffs = coeffs

    R = corr_func(data)

    @lru_cache(maxsize=None)
    def func(x):
        if x < 0:
            return func(-x)
        if x <= m + n:
            return R(x)

        rs = np.array([func(x - i) for i in range(1, m + 1)])

        return rs.dot(ar_coeffs)

    def res(x):
        return func(x) / R(0)

    return res


def model_arma(m, n, data, eps=1e-5, maxiter=100):

    if m == 0:
        return model_ma(n, data, eps, maxiter)
    if n == 0:
        return model_ar(m, data)

    coeffs = calc_model_arma(m, n, data, eps, maxiter)

    ar_coeffs, ma_coeffs = coeffs

    def get_model(xs, x0):
        if len(xs) != m:
            raise ValueError('Length of `xs` must be equal to {}'.format(m))

        xs = np.array(xs)
        w = np.random.normal(size=(n + 1))

        x = x0

        while True:
            yield x

            x = xs.dot(ar_coeffs) + ma_coeffs.dot(w)

            w[1:] = w[:-1]
            w[0] = np.random.normal()

            xs[1:] = xs[:-1]
            xs[0] = x

    return get_model
