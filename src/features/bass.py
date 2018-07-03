import math
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage
import src.utils as utils


def extract_bass_lfs(x, sr, hop_length=176, n_fft=1408, plot=False):
    s = abs(librosa.stft(x, n_fft, hop_length))
    s = s[1:11, :]
    f = librosa.fft_frequencies(sr, n_fft)[1:11]

    # Clip to 9th decile
    p = np.percentile(s, 90)
    s[s > p] = p

    if plot:
        librosa.display.specshow(s, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='coolwarm')

    return s, f


# x, sr = librosa.load('../../data/raw/dream_on.wav')
# q, f = extract_bass_lfs(x[:int(len(x) / 10)], sr, plot=True)
# plt.show()
