import math
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage

import src.utils as utils


def aggregate_pitch(x, bins_per_octave=36, new_bins_per_octave=36):
    res = np.zeros([new_bins_per_octave, x.shape[1]])

    octaves = int(x.shape[0] / bins_per_octave)
    samples_per_pitch = max(int(bins_per_octave / new_bins_per_octave), 1)
    for p in range(new_bins_per_octave):
        rows = []
        for octave in range(octaves):
            row = p * samples_per_pitch + octave * bins_per_octave
            offset = row - samples_per_pitch/2
            rows += [utils.round(offset + sample) for sample in range(samples_per_pitch) if utils.round(offset + sample) >= 0]

        res[p] = np.sum(x[rows, :], axis=0)

    return res


def tune(x, bins_per_octave=36):
    bins_per_key = bins_per_octave / 12
    res = np.zeros([12, x.shape[1]])

    for k in range(12):
        offset = int(k * bins_per_key)
        alfa = x[offset, :]
        beta = x[offset + 1, :]
        gamma = x[offset + 2, :]

        sums = alfa + beta + gamma
        ps = alfa/sums * -1 + gamma/sums * 1
        p = np.sum(ps) / x.shape[1]

        asg = np.subtract(alfa, gamma)
        res[k] = beta - 0.25 * asg * p
        # res[k] = alfa + beta + gamma

    return res


def extract_harmony_chroma(x, sr, f_min=73.41619, bins_per_octave=36, octaves=3, hop_length_ms=92.9, plot=False):
    n_bins = utils.round(bins_per_octave * octaves)
    hop_length = utils.hop_length_from_ms(hop_length_ms, sr)

    # We shift the bins a bit down so the key itslef is the middle bin instead of first
    margin = (bins_per_octave/12+1)/2
    f_start = f_min
    f_min = (f_min/2/bins_per_octave * (bins_per_octave - margin)) + f_min/2

    q = librosa.cqt(x, sr=sr, fmin=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length)
    f = librosa.cqt_frequencies(n_bins=n_bins, fmin=f_min, bins_per_octave=bins_per_octave)

    pb = aggregate_pitch(abs(q), bins_per_octave, bins_per_octave)  # Sum octaves into pitch classes. 3 bins per pitch
    tuned = tune(pb, bins_per_octave)  # Tune the pitch bins onto one
    filtered = scipy.ndimage.median_filter(tuned, (1, 6))  # Apply a median filter to smoothen things

    if plot:
        librosa.display.specshow(filtered, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note', bins_per_octave=12, fmin=f_start,
                                 cmap='coolwarm')

    return q, f


# x, sr = librosa.load('../../data/raw/dream_on.wav')
# q, f = extract_harmony_chroma(x[:int(len(x) / 10)], sr, plot=True)
# plt.show()
