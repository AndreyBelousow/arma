import numpy as np
import matplotlib.pyplot as plt
import math


def correlation_function(x, n, k, average):
    q = 1 / (n - k - 1)

    i = 0
    sum = 0
    while i < n - k:
        sum += (x[i] - average) * (x[i + k] - average)
        i += 1

    res = q * sum
    return res


def main():
    y = np.fromfile('source.txt', dtype=float, sep=' ')
    n = y.size
    x = np.linspace(0, n - 1, n)

    min = np.min(y)
    max = np.max(y)
    average = np.average(y)
    variance = np.sum(np.power(y - average, 2)) / (n - 1)
    standard_deviation = np.sqrt(variance)

    plt.plot(x[0:150], y[0:150], label='source process')
    plt.plot(x[0:150], np.full(150, average), label='average')
    plt.plot(x[0:150], np.full(150, average + standard_deviation), 'g',
             np.full(150, average - standard_deviation), 'g',
             label='standard deviation')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()

    print("avg: ", average)
    print("variance: ", variance)
    print("std_dev: ", standard_deviation)

    k = 0
    m = 11
    R = np.zeros(m)
    r = np.zeros(m)

    while k <= m - 1:
        R[k] = correlation_function(y, n, k, average)
        r[k] = R[k] / variance
        k += 1

    interval = m - 1
    while abs(r[interval]) <= math.exp(-1):
        interval -= 1

    eps = np.exp(-1)
    # show plot of source normalized correlation
    plt.plot(np.linspace(0, m - 1, m), r, label='source normalized correlation')
    plt.plot(np.linspace(0, m - 1, m), np.full(m, eps), 'g', label='+exp^-1')
    plt.plot(np.linspace(0, m - 1, m), np.full(m, -eps), 'g', label='-exp^-1')
    plt.plot([interval, interval], [np.min(r), np.max(r)], 'r', linestyle='--', label='correlation interval')
    plt.xticks(np.linspace(0, m - 1, m))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()

    print("Correlation:")
    for i in range(m):
        print("    ", round(R[i], 4))

    print("Normalized correlation:")
    for i in range(m):
        print("    ", round(r[i], 4))

    print("Correlation interval: ", interval)


if __name__ == '__main__':
    main()
