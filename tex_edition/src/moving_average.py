import numpy as np
from correlation import *


def calc_model_ma0(data):
    """
        R(0) = a_0**2

        Exact solution:
        a_0 = sqrt(R(0))

        R - correlation function
        a_i - model coef.
    """
    data = np.array(data)

    R = corr_func(data)

    R0 = R(0)

    if R0 < 0:
        raise ArithmeticError(
            'Dispersion should be non-negative, maybe bug in corr_func')

    return np.sqrt([R0])


def theory_ncf_ma0(data):
    r = normalized_corr_func(data)

    def func(n):
        if n == 0:
            return r(0)
        return 0

    return func


def model_ma0(data):
    """
        x(t) = w(t) * a_0

        w - white noise
        a_i - coefs. from calc_model_ma0
    """
    a = calc_model_ma0(data)

    def get_model(x0):
        w = np.random.normal()
        x = x0

        while True:
            yield x

            x = w * a
            w = np.random.normal()

    return get_model


def calc_model_ma1(data, eps=1e-5, maxiter=100):
    """
        R(0) = a_0**2 + a_1**2
        R(1) = a_0 * a_1

        Iterative solution:
        a_0 = R(1) / a_1
        a_1 = sqrt(R(0) - a_0**2)

        R - correlation function
        a_i - model coefs.
    """
    data = np.array(data)

    R = corr_func(data)

    R0 = R(0)
    R1 = R(1)

    a0 = 0
    a1 = R0

    for _ in range(maxiter):
        if abs(R0 - a0**2 - a1**2) < eps \
                and abs(R1 - a0 * a1) < eps:
            return np.array([a0, a1])

        a0 = R1 / a1

        if (R0 - a0**2) < 0:
            raise ArithmeticError('Solution doesn\'t exist [ma(1)]')

        a1 = np.sqrt(R0 - a0**2)

    if abs(R0 - a0**2 - a1**2) > 10 * eps \
            or abs(R1 - a0 * a1) > 10 * eps:
        raise ArithmeticError('Didn\'t coverage to solution')

    return np.array([a0, a1])


def theory_ncf_ma1(data):
    r = normalized_corr_func(data)

    def func(n):
        if abs(n) <= 1:
            return r(n)
        return 0

    return func


def model_ma1(data, eps=1e-5, maxiter=100):
    """
        x(t) = a_0 * w(t) + a_1 * w(t - 1)

        w - white noise
        a_i - coefs. from calc_model_ma1
    """
    a = calc_model_ma1(data, eps, maxiter)

    def get_model(x0):
        w = np.random.normal(size=2)

        x = x0

        while True:
            yield x

            x = a.dot(w)

            w[1:] = w[:-1]
            w[0] = np.random.normal()

    return get_model


def calc_model_ma2(data, eps=1e-5, maxiter=100):
    """
        R(0) = a_0**2 + a_1**2 + a_2**2
        R(1) = a_0 * a_1 + a_1 * a_2
        R(2) = a_0 * a_2

        Iterative solution:
        a_2 = R(2) / a_0
        a_1 = (R(1) - a_1 * a_2) / a_0
        a_0 = sqrt(R(0) - a_1**2 - a_2**2) if R(0) >= a_1**2 + a_2**2

        R - correlation function
        a_i - model coeffs.
    """
    data = np.array(data)

    R = corr_func(data)

    R0 = R(0)
    R1 = R(1)
    R2 = R(2)

    a0 = np.sqrt(R0)
    a1 = 0
    a2 = 0

    for _ in range(maxiter):
        if abs(R0 - a0**2 - a1**2 - a2**2) < eps \
                and abs(R1 - a0 * a1 - a1 * a2) < eps \
                and abs(R2 - a0 * a2) < eps:
            return np.array([a0, a1, a2])

        a2 = R2 / a0
        a1 = (R1 - a1 * a2) / a0

        if R0 < a1 ** 2 + a2 ** 2:
            raise ArithmeticError('Solution doesn\'t exist [ma(2)]')

        a0 = np.sqrt(R(0) - a1**2 - a2**2)

    if abs(R0 - a0**2 - a1**2 - a2**2) > 10 * eps \
            or abs(R1 - a0 * a1 - a1 * a2) > 10 * eps \
            or abs(R2 - a0 * a2) > 10 * eps:
        raise ArithmeticError('Didn\'t coverage to solution')

    return np.array([a0, a1, a2])


def theory_ncf_ma2(data):
    r = normalized_corr_func(data)

    def func(n):
        if abs(n) <= 2:
            return r(n)
        return 0

    return func


def model_ma2(data, eps=1e-5, maxiter=100):

    a = calc_model_ma2(data, eps, maxiter)

    def get_model(x0):
        x = x0
        w = np.random.normal(size=3)

        while True:
            yield x

            x = a.dot(w)

            w[1:] = w[:-1]
            w[0] = np.random.normal()

    return get_model


def calc_model_ma3(data, eps=1e-5, maxiter=100):

    data = np.array(data)

    R = corr_func(data)

    R0 = R(0)
    R1 = R(1)
    R2 = R(2)
    R3 = R(3)

    a0 = np.sqrt(R0)
    a1 = 0
    a2 = 0
    a3 = 0

    for _ in range(maxiter):
        if abs(R0 - a0**2 - a1**2 - a2**2 - a3**2) < eps \
                and abs(R1 - a0 * a1 - a1 * a2 - a2 * a3) < eps \
                and abs(R2 - a0 * a2 - a1 * a3) < eps \
                and abs(R3 - a0 * a3) < eps:
            return np.array([a0, a1, a2, a3])

        a3 = R3 / a0
        a2 = (R2 - a1 * a3) / a0
        a1 = (R1 - a1 * a2 - a2 * a3) / a0

        if R0 < a1**2 + a2**2 + a3**2:
            raise ArithmeticError('Solution doesn\'t exist [ma(3)]')

        a0 = np.sqrt(R0 - a1**2 - a2**2 - a3**2)

    if abs(R0 - a0**2 - a1**2 - a2**2 - a3**2) > 10 * eps \
            or abs(R1 - a0 * a1 - a1 * a2 - a2 * a3) > 10 * eps \
            or abs(R2 - a0 * a2 - a1 * a3) > 10 * eps \
            or abs(R3 - a0 * a3) > 10 * eps:
        raise ArithmeticError('Didn\'t coverage to solution')

    return np.array([a0, a1, a2, a3])


def theory_ncf_ma3(data):
    r = normalized_corr_func(data)

    def func(n):
        if abs(n) <= 3:
            return r(n)
        return 0

    return func


def model_ma3(data, eps=1e-5, maxiter=100):

    a = calc_model_ma3(data, eps, maxiter)

    def get_model(x0):
        x = x0
        w = np.random.normal(size=4)

        while True:
            yield x

            x = a.dot(w)

            w[1:] = w[:-1]
            w[0] = np.random.normal()

    return get_model


def calc_model_ma(n, data, eps=1e-5, maxiter=100):

    if n == 0:
        return calc_model_ma0(data)
    if n == 1:
        return calc_model_ma1(data, eps, maxiter)
    if n == 2:
        return calc_model_ma2(data, eps, maxiter)
    if n == 3:
        return calc_model_ma3(data, eps, maxiter)

    raise ValueError('Calculating coeffs. for `n > 3` not implemented')


def theory_ncf_ma(n, data):

    if n == 0:
        return theory_ncf_ma0(data)
    if n == 1:
        return theory_ncf_ma1(data)
    if n == 2:
        return theory_ncf_ma2(data)
    if n == 3:
        return theory_ncf_ma3(data)

    raise ValueError('NCF for `n > 3` not implemented')


def model_ma(n, data, eps=1e-5, maxiter=100):

    if n == 0:
        return model_ma0(data)
    if n == 1:
        return model_ma1(data, eps, maxiter)
    if n == 2:
        return model_ma2(data, eps, maxiter)
    if n == 3:
        return model_ma3(data, eps, maxiter)

    raise ValueError('Models with `n > 3` not implemented')
