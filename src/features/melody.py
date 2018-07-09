import math
import numpy as np
import librosa, librosa.display
import src.utils as utils


def average_bins(q, bins_per_octave=96, cutdown=96):
    qa = np.zeros_like(q)
    for w in range(cutdown, q.shape[0] - cutdown):
        Jw = int((q.shape[0] - w) / bins_per_octave)
        js = [w + j * bins_per_octave for j in range(Jw)]
        qa[w] = np.sum(q[js, :], axis=0) / (Jw + 1)

    return qa[cutdown:q.shape[0] - cutdown, :]


def extract_melody_cqt(x, sr, f_min=195.9977, bins_per_octave=96, octaves=5, hop_length_ms=11.6, plot=False):
    n_bins = bins_per_octave * octaves + 16
    hop_length = utils.hop_length_from_ms(hop_length_ms, sr)

    q = librosa.cqt(x, sr=sr, fmin=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length)
    f = librosa.cqt_frequencies(n_bins=n_bins, fmin=f_min, bins_per_octave=bins_per_octave)

    # Average bins
    qa = average_bins(abs(q), bins_per_octave, bins_per_octave)

    # Represent logarithmic
    lq = np.log(abs(qa) + 1)

    # Cutdown low values / noise
    third_quartile = np.percentile(lq, 25)
    lq[lq <= third_quartile] = 0

    if plot:
        fstart = f[bins_per_octave]
        librosa.display.specshow(lq, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave, fmin=fstart,
                                 cmap='coolwarm')

    return qa, f[bins_per_octave: -bins_per_octave]


# import matplotlib.pyplot as plt
# x, sr = librosa.load('../../data/raw/dream_on.wav')
# q, f = extract_melody_cqt(x[:int(len(x) / 10)], sr, plot=True)
# print(q.shape)
# plt.show()
