import math
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage
import src.utils as utils


def local_mean(dS, M):
    mu = np.empty_like(dS)
    mu[0] = 0.
    # First and last part: if n < m or (N - n) > m
    for n in range(1, M):
        mu[n] = dS[:n * 2].sum() * 1 / (2 * n + 1)
        mu[-n] = dS[-n * 2:].sum() * 1 / (2 * n + 1)
    # Middle part
    for n in range(M, len(dS) - M + 1):
        mu[n] = dS[n - M:n + M].sum() * 1 / (2 * M + 1)
    return mu


def extract_rhythm_odf(x, sr, plot=False):
    hop_length = utils.round(11.6 / 1000 * sr)
    n_fft = utils.round(23.2 / 1000 * sr)

    s = librosa.stft(x, n_fft, hop_length)
    S = librosa.amplitude_to_db(abs(s))
    # S = np.log(abs(S) + 1)
    f = librosa.fft_frequencies(sr, n_fft)

    onset_bands = librosa.onset.onset_strength_multi(S=S, sr=sr, channels=[0, 4, 13, s.shape[0]])

    for band in onset_bands:
        # Clip to 9th decile
        band -= local_mean(band, n_fft)
        band[band < 0] = 0.

        p = np.percentile(band, 90)
        band[band > p] = p

    if plot:
        labels = ['L', 'M', 'H']
        for i in range(onset_bands.shape[0]):
            t = librosa.frames_to_time(onset_bands[i], sr, hop_length, n_fft)
            plt.subplot(onset_bands.shape[0], 1, i + 1)
            plt.plot(onset_bands[i])
            plt.title(labels[i])

    return s, f


# x, sr = librosa.load('../../data/raw/dream_on.wav')
# q, f = extract_rhythm_odf(x[:int(len(x) / 10)], sr, plot=True)
# plt.show()
