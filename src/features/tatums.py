import librosa, librosa.display
import matplotlib.pyplot as plt


def extract_tatums(x, sr, tatum_size=16, plot=False):
    t = librosa.beat.tempo(x, sr, start_bpm=70.)
    t *= tatum_size
    tatum_length_ms = 60000/t
    samples_per_tatum = tatum_length_ms * sr
    print(tatum_length_ms)
    print(samples_per_tatum)
    return tatum_length_ms


x, sr = librosa.load('../../data/raw/dream_on.wav')
t = extract_tatums(x, sr, plot=True)
plt.show()
