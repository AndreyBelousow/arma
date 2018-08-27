def correlation_function(x, n, k, average):
    q = 1 / (n - k - 1)

    i = 0
    sum = 0
    while i < n - k:
        sum += (x[i] - average) * (x[i + k] - average)
        i += 1

    res = q * sum
    return res
