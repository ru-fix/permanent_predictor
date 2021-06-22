import numpy as np


def linspace(start, stop, num=1, dtype=float):
    return np.linspace(start, stop, num, dtype=dtype)


def logspace(start, stop, num=1, dtype=float):
    return np.logspace(np.log10(start), np.log10(stop), num, dtype=dtype)


def statspace(start, num=1, dtype=float):
    return np.repeat(np.array([start], dtype=dtype), num)
