import math


def round(x):
    """
    Fixes the completely idiotic way of rounding python uses by default
    :param x:
    :type x:
    :return:
    :rtype:
    """
    return int(x + 0.5)


def hop_length_from_ms(t, sr):
    return int(math.pow(2, round(math.log2(int(sr / 1000 * t)))))
