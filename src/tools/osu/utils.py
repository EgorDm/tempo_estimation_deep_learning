from bisect import bisect_left


def original_type(s):
    if is_int(s):
        return int(s)
    elif is_float(s):
        return float(s)
    else:
        return s


def is_int(s):
    return is_type(s, int)


def is_float(s):
    return is_type(s, float)


def is_type(s, val_type):
    try:
        val_type(s)
        return True
    except ValueError:
        return False


def clip(val, min_, max_):
    return min_ if val < min_ else max_ if val > max_ else val


def binary_search(a, x, lo=0, hi=None):
    hi = hi if hi is not None else len(a)
    pos = bisect_left(a, x, lo, hi)
    return pos if pos != hi and a[pos] == x else ~pos
