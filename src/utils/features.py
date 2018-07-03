import math


def hop_length_from_ms(t, sr):
    return int(math.pow(2, round(math.log2(int(sr / 1000 * t)))))
