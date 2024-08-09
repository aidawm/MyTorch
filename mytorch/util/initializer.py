import numpy as np


def xavier_initializer(shape):
    n = shape[0]
    lower = -(1 / np.sqrt(n))
    upper = 1 / np.sqrt(n)
    return np.random.uniform(low=lower, high=upper, size=shape)


def he_initializer(shape):
    n = shape[0]
    std = np.sqrt(2.0 / n)
    return np.random.normal(0.0, std, shape)


def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    return np.random.normal(mean, stddev, shape)


def zero_initializer(shape):
    return np.zeros(shape, dtype=np.float64)


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)


def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
