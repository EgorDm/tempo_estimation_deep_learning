import glob
import os
import random
from typing import Tuple
import soundfile as sf
import numpy as np


class Batcher:
    def __init__(self, datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True, repeat_dataset=True):
        self.buffer_size = max(buffer_size, batch_size * 2)
        self.samples = []
        self.cursor = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.randomize = randomize
        self.repeat_dataset = repeat_dataset

        # Load the dataset entries
        if sample_files is None: sample_files = []

        if datasets is not None:
            if isinstance(datasets, str): datasets = [datasets]
            for dataset in datasets: sample_files += self._load_dataset_entries(dataset)

        if len(sample_files) == 0: raise Exception(f'Not enough data in given datasets.')
        if self.randomize: random.shuffle(sample_files)
        self.train_data_files = sample_files

        self._buffer_batches()

    def _load_dataset_entries(self, dataset):
        if not os.path.exists(dataset): raise Exception(f'Dataset not found {dataset}')
        dataset = dataset.replace('\\', '/')

        files = glob.glob(f'{dataset}/*.wav')
        return ['.'.join(file.replace('\\', '/').split('.')[:-1]) for file in files]

    def _buffer_batches(self):
        """
        Loads new batches into the batch buffer
        """
        tmp = self.samples
        self.samples = []

        while len(self.samples) < self.buffer_size:
            self.cursor += 1
            if self.cursor >= len(self.train_data_files):
                if not self.repeat_dataset: return
                self.cursor = 0
                self.epoch += 1

            entry = self.train_data_files[self.cursor]
            self._create_samples_from_entry(entry)

        if self.randomize: random.shuffle(self.samples)

        # We push the leftovers to front since they had their share of suffling and we want to empty cache asap
        self.samples += tmp

    def _create_samples_from_entry(self, entry): pass

    def _post_process_sample(self, sample) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post process the batch before returning it to the model
        """
        return sample

    def get_batch(self, count=None) -> Tuple[np.ndarray, np.ndarray]:
        if count is None: count = self.batch_size
        if len(self.samples) < count: self._buffer_batches()  # Buffer batches if there are not enough
        count = min(len(self.samples), count)

        inputs = []
        labels = []
        for samples in self.samples[-count:]:
            i, l = self._post_process_sample(samples)
            inputs.append(i)
            labels.append(l)

        self.samples = self.samples[:-count]
        return np.stack(inputs, axis=0), np.stack(labels, axis=0)

    def generator(self):
        while 1:
            x, y = self.get_batch()
            if len(self.samples) == 0: break
            yield x, y


class CacheableBatcher(Batcher):
    """
    Requires every batch to specify the cache position as the first element
    """

    def __init__(self, datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True):
        self._cached_features = {}

        super().__init__(datasets, sample_files, buffer_size, batch_size, randomize)

    def _buffer_batches(self):
        # Clean cache
        remove_cached = list(self._cached_features.keys())
        for batch in self.samples:
            if len(remove_cached) == 0: break
            if batch[0] in remove_cached: remove_cached.remove(batch[0])

        for k in remove_cached: self._cached_features.pop(k)

        # Default behaviour
        super()._buffer_batches()

    def _get_sample_data(self, sample) -> np.ndarray:
        pass

    def _get_sample_label(self, sample) -> np.ndarray:
        pass

    def _post_process_sample(self, sample) -> Tuple[np.ndarray, np.ndarray]:
        i = self._get_sample_data(sample)
        l = self._get_sample_label(sample)
        return i, l

    def _add_to_cache(self, d):
        key = random.randint(0, 9999999)  # Lets rely on randomness 🤷 to pick a unique cache spot
        self._cached_features[key] = d
        return key


class AudioFeatureBatcher(CacheableBatcher):
    def __init__(self, feature, sampling_mode='stratisfied', datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True):
        self._window_size = 85
        self._label_subdivisions = 3  # Oversample the negatves class by a ratio 2:1
        self._sampling_mode = sampling_mode.lower()
        self._feature = feature

        super().__init__(datasets, sample_files, buffer_size, batch_size, randomize)

    def _create_samples_from_entry(self, entry):
        # Load data
        with open(f'{entry}.txt', 'r') as f: labels = f.readlines()
        x, sr = sf.read(f'{entry}.wav')

        q, _ = self._feature(x, sr)
        q = np.transpose(q, [1, 0])  # Swap the temporal and bin axes. [temporal, bins]

        cache_key = self._add_to_cache(q)
        position_to_frame = q.shape[0] / len(x)
        half_window = self._window_size / 2

        def create_sample(centre, label):
            frame_start = int(centre - half_window)
            frame_end = int(centre + half_window)
            if frame_start < half_window or frame_end > q.shape[0] - half_window: return
            self.samples.append((cache_key, frame_start, frame_end, label))

        labels = [float(label.split(' ')[0]) for label in labels]  # TODO: here we can add some constraints

        # Full mode creates samples for every position in audio. Dont use it for training since labels are dispropotionate
        if self._sampling_mode == 'full':
            label_positions = [int(label * sr * position_to_frame) for label in labels]
            for i in range(len(q)):  create_sample(i, 1 if i in label_positions else 0)

        # Creates only positives and a balanced amount of negatives spaced between the positives
        elif self._sampling_mode == 'stratisfied':
            # Create positives
            for label in labels: create_sample(int(label * sr * position_to_frame), 1)

            # Create negatives (just pick some pos iN between)
            for i in range(len(labels) - 1):
                position_start = int(labels[i] * sr * position_to_frame)
                position_end = int(labels[i + 1] * sr * position_to_frame)
                if (position_end - position_start) <= self._label_subdivisions: continue

                step_size = (position_end - position_start) / self._label_subdivisions
                for j in range(1, self._label_subdivisions): create_sample(position_start + step_size * j, 0)

    def _get_sample_data(self, sample) -> np.ndarray:
        return self._cached_features[sample[0]][sample[1]:sample[2], :]

    def _get_sample_label(self, sample) -> np.ndarray:
        return np.array([1, 0]) if sample[3] == 1 else np.array([0, 1])

# import src.features as features
# batcher = AudioFeatureBatcher(features.extract_melody_cqt, datasets=['../../../data/processed/default/train'], buffer_size=10000)
# for i in range(10):
#     test = batcher.get_batches()
