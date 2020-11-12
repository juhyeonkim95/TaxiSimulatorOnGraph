import numpy as np


def lerp(a, b, x):
    return a * (1-x) + b * x


def clamp(x, max_x, min_x):
    if x > max_x:
        return max_x
    elif x < min_x:
        return min_x
    return x


def softmax(x, k) :
    c = np.max(x)
    exp_a = np.exp(k * (x-c))
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softmax_pow(x, k):
    c = x / np.max(x)
    c = c ** k
    return c / np.sum(c)