import soundfile as sf
import numpy as np
from src.tools.data.batcher import CacheableBatcher
from src.utils.iteration import Iterator
import random


class AudioFeatureBatcher(CacheableBatcher):
    def __init__(self, feature, sampling_mode='stratisfied', datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True):
        self._window_size = 85
        self._sampling_mode = sampling_mode.lower()
        self._feature = feature

        super().__init__(datasets, sample_files, buffer_size, batch_size, randomize)

    # noinspection PyUnresolvedReferences
    def _create_samples_from_entry(self, entry):
        # Load data
        with open(f'{entry}.txt', 'r') as f: labels = f.readlines()
        tempo_changes, beats = self.parse_labels(labels)
        x, sr = sf.read(f'{entry}.wav')

        # Extract features
        q, _ = self._feature(x, sr)
        q = np.transpose(q, [1, 0])  # Swap the temporal and bin axes. [temporal, bins]

        cache_key = self._add_to_cache(q)
        position_to_frame = q.shape[0] / len(x)
        half_window = self._window_size / 2
        tatum_size = 16  # 1/16 note is fine i guess.
        duration = len(x) / sr

        # Create samples
        def create_sample(centre, label):
            frame_start = int(centre - half_window)
            frame_end = int(centre + half_window)
            if frame_start < half_window or frame_end > q.shape[0] - half_window: return
            self.samples.append((cache_key, frame_start, frame_end, label))

        # Full mode creates samples for every position in audio. Dont use it for training since labels are dispropotionate
        if self._sampling_mode == 'full':
            def next_tempo(it, tempo):
                it.half_tatum_length = tempo[1] * tempo[2] / 1000 / tatum_size / 2
                it.time = tempo[0] + it.half_tatum_length

            timing_it = Iterator(tempo_changes, next_callback=next_tempo)
            beat_it = Iterator(beats)

            while timing_it.time < duration:
                if timing_it.peek() is not None and timing_it.peek()[0] <= timing_it.time: timing_it.next()
                current_beat = None
                if beat_it.peek() is not None and beat_it.peek()[0] <= timing_it.time: current_beat = beat_it.next()

                mid_tatum = timing_it.time - timing_it.half_tatum_length + random.uniform(-0.5, 0.5) * timing_it.half_tatum_length
                position = mid_tatum * sr * position_to_frame

                create_sample(position, 1 if current_beat is not None else 0)
                timing_it.time += timing_it.half_tatum_length * 2
        # Randomize the samples.
        elif self._sampling_mode == 'random':
            def next_tempo(it, tempo): it.half_tatum_length = tempo[1] * tempo[2] / 1000 / tatum_size / 2

            def next_beat(it, beat): it.time = beat[0]

            timing_it = Iterator(tempo_changes, next_callback=next_tempo)
            beat_it = Iterator(beats, next_callback=next_beat)

            while beat_it.next() is not None:
                while timing_it.peek() is not None and timing_it.peek()[0] <= beat_it.time: timing_it.next()

                mid_tatum = beat_it.time + random.uniform(-0.5, 0.5) * timing_it.half_tatum_length
                position = mid_tatum * sr * position_to_frame

                create_sample(position, 1)

                # Pick a random negative between the two labeled tantums
                if beat_it.peek() is not None:
                    position_start = int((beat_it.current()[0] + timing_it.half_tatum_length * 2) * sr * position_to_frame)
                    position_end = int((beat_it.peek()[0] - timing_it.half_tatum_length * 2) * sr * position_to_frame)
                    create_sample(position_start + random.uniform(0, 1) * (position_end - position_start), 0)

    def _get_sample_data(self, sample) -> np.ndarray:
        return self._cached_features[sample[0]][sample[1]:sample[2], :]

    def _get_sample_label(self, sample) -> np.ndarray:
        return np.array([1, 0]) if sample[3] == 1 else np.array([0, 1])

    def parse_labels(self, lines, beat_picker=None):
        tempo_count = int(lines[0])

        tempo_changes = []
        for l in lines[1:tempo_count + 1]:
            tokens = l.split(' ')
            tempo_changes.append((float(tokens[0]), float(tokens[1]), int(tokens[2])))

        beats = []
        for l in lines[tempo_count + 2:]:
            tokens = l.split(' ')
            beat = (float(tokens[0]), int(tokens[1]), bool(tokens[2]))
            if beat_picker is not None and not beat_picker(beat): continue
            if beat[1] != 1: continue
            # if not beat[2]: continue
            beats.append(beat)

        return tempo_changes, beats

# batcher = AudioFeatureBatcher(features.extract_melody_cqt, datasets=['../../../data/processed/default/train'], sampling_mode='random', buffer_size=1000)
# test = batcher.get_batch()
