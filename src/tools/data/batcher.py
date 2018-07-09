import glob
import os
import random
import librosa
import soundfile as sf
import src.features as features
import numpy as np


class Batcher:
    def __init__(self, datasets, buffer_size=10, validate_size=0.1, batch_count=20) -> None:
        self.buffer_size = max(buffer_size, batch_count * 2)
        self.batches = []
        self.cursor = 0
        self.epoch = 0
        self.batch_count = batch_count

        # Load the dataset entries
        if isinstance(datasets, str): datasets = [datasets]

        data_files = []
        for dataset in datasets:
            if not os.path.exists(dataset): raise Exception(f'Dataset not found {dataset}')
            dataset = dataset.replace('\\', '/')

            files = glob.glob(f'{dataset}/*.wav')
            data_files += ['.'.join(file.replace('\\', '/').split('.')[:-1]) for file in files]

        if len(data_files) == 0: raise Exception(f'Not enough data in given datasets.')

        # Split the data in training and validation
        random.shuffle(data_files)
        validate_split = max(int(len(data_files) * validate_size), 1)
        self.train_data_files = data_files[validate_split:]
        self.validate_data_files = data_files[:validate_split]

        self._buffer_batches()

    def _buffer_batches(self):
        tmp = self.batches
        self.batches = []

        while len(self.batches) < self.buffer_size:
            self.cursor += 1
            if self.cursor >= len(self.train_data_files):
                self.cursor = 0
                self.epoch += 1

            sample = self.train_data_files[self.cursor]
            x, sr = sf.read(f'{sample}.wav')
            with open(f'{sample}.txt', 'r') as f: labels = f.readlines()
            self._create_batches((x, sr), labels)
        random.shuffle(self.batches)

        # We push the leftovers to front since they had their share of suffling and we want to empty cache asap
        self.batches += tmp

    def _create_batches(self, audio, labels):
        self.batches += [i for i in range(10)]

    def _post_process(self, batch):
        return batch

    def get_batches(self, count=None):
        if count is None: count = self.batch_count
        if len(self.batches) < count:
            self._buffer_batches()

        inputs = []
        labels = []
        for batch in self.batches[-count:]:
            i, l = self._post_process(batch)
            inputs.append(i)
            labels.append(l)

        self.batches = self.batches[:-count]
        return np.stack(inputs, axis=0), np.stack(labels, axis=0)


class MelodyBatcher(Batcher):
    _window_size = 85
    _cached_features = {}
    _label_subdivisions = 3  # Oversample the negatves class by a ratio 2:1

    def _buffer_batches(self):
        # Clean cache
        remove_cached = list(self._cached_features.keys())
        for i in range(len(self.batches)):
            if len(remove_cached) == 0: break
            if self.batches[i][0] in remove_cached: remove_cached.remove(self.batches[i][0])

        for k in remove_cached:
            self._cached_features.pop(k)

        # Default behaviour
        super()._buffer_batches()

    def _create_batches(self, audio, labels):
        q, _ = features.extract_melody_cqt(audio[0], audio[1])
        cache_idx = random.randint(0, 9999999)  # Lets rely on randomness 🤷
        self._cached_features[cache_idx] = q
        position_to_frame = q.shape[1] / len(audio[0])
        half_window = self._window_size / 2

        def create_sample(arr, centre, label):
            frame_start = int(centre - half_window)
            frame_end = int(centre + half_window)
            if frame_start < half_window or frame_end > q.shape[1] - half_window: return
            arr.append((cache_idx, frame_start, frame_end, label))

        # Create positives
        for label in labels:
            position = int(float(label.split(' ')[0]) * audio[1] * position_to_frame)
            create_sample(self.batches, position, 1)

        # Create negatives
        for i in range(len(labels) - 1):
            position_start = int(float(labels[i].split(' ')[0]) * audio[1] * position_to_frame)
            position_end = int(float(labels[i + 1].split(' ')[0]) * audio[1] * position_to_frame)
            if (position_end - position_start) <= self._label_subdivisions: continue

            step_size = (position_end - position_start) / self._label_subdivisions
            for j in range(1, self._label_subdivisions):
                create_sample(self.batches, position_start + step_size * j, 0)

    def _post_process(self, batch):
        i = self._cached_features[batch[0]][:, batch[1]:batch[2]]
        l = np.array([1, 0]) if batch[3] == 1 else np.array([0, 1])
        return i, l


batcher = MelodyBatcher(['../../../data/processed/default'], buffer_size=10000)