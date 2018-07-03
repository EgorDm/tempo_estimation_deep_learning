import numpy as np


def quadratic_interpolate(alfa: np.ndarray, beta: np.ndarray, gamma: np.ndarray):
    """
    Quadratic interpolation
    http://www.dtic.upf.edu/~ggeiger/InfoAudioMusica/lab-5.html
    :param alfa: K-1 bin
    :type alfa:
    :param beta: K0 bin
    :type beta:
    :param gamma: K1 bin
    :type gamma:
    :return:
    :rtype:
    """
    asg = np.subtract(alfa, gamma)
    p = 0.5 * asg/(alfa - 2*beta + gamma)
    y = beta - 0.25 * asg * p
    return p, y