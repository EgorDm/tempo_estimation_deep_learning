import glob
import os
import random
import soundfile as sf
import src.features as features
import numpy as np


class Batcher:
    def __init__(self, datasets, buffer_size=10, batch_size=20, randomize_batches=True):
        self.buffer_size = max(buffer_size, batch_size * 2)
        self.batches = []
        self.cursor = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.randomize_batches = randomize_batches

        # Load the dataset entries
        if isinstance(datasets, str): datasets = [datasets]

        data_files = []
        for dataset in datasets: data_files += self._load_dataset_entries(dataset)
        if len(data_files) == 0: raise Exception(f'Not enough data in given datasets.')

        random.shuffle(data_files)
        self.train_data_files = data_files

        self._buffer_batches()

    def _load_dataset_entries(self, dataset):
        if not os.path.exists(dataset): raise Exception(f'Dataset not found {dataset}')
        dataset = dataset.replace('\\', '/')

        files = glob.glob(f'{dataset}/*.wav')
        return ['.'.join(file.replace('\\', '/').split('.')[:-1]) for file in files]

    def _buffer_batches(self):
        """
        Loads new batches into the batch buffer
        :return:
        :rtype:
        """
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
            self._create_batches({'data': x, 'sr': sr}, labels)

        if self.randomize_batches: random.shuffle(self.batches)

        # We push the leftovers to front since they had their share of suffling and we want to empty cache asap
        self.batches += tmp

    def _create_batches(self, audio, labels):
        """
        Create batches from a dataset sample
        :param audio:
        :type audio: dict
        :param labels:
        :type labels:
        :return:
        :rtype:
        """
        pass

    def _post_process(self, batch):
        """
        Post process the batch before returning it to the model
        :param batch:
        :type batch:
        :return:
        :rtype:
        """
        return batch

    def get_batches(self, count=None):
        if count is None: count = self.batch_size
        if len(self.batches) < count: self._buffer_batches()  # Buffer batches if there are not enough

        inputs = []
        labels = []
        for batch in self.batches[-count:]:
            i, l = self._post_process(batch)
            inputs.append(i)
            labels.append(l)

        self.batches = self.batches[:-count]
        return np.stack(inputs, axis=0), np.stack(labels, axis=0)

    def generator(self):
        while 1:
            x, y = self.get_batches()
            yield x, y


class MelodyBatcher(Batcher):
    def __init__(self, datasets, buffer_size=10, batch_size=20, randomize_batches=True) -> None:
        self._window_size = 85
        self._cached_features = {}
        self._label_subdivisions = 3  # Oversample the negatves class by a ratio 2:1

        super().__init__(datasets, buffer_size, batch_size, randomize_batches)

    def _buffer_batches(self):
        # Clean cache
        remove_cached = list(self._cached_features.keys())
        for batch in self.batches:
            if len(remove_cached) == 0: break
            if batch[0] in remove_cached: remove_cached.remove(batch[0])

        for k in remove_cached: self._cached_features.pop(k)

        # Default behaviour
        super()._buffer_batches()

    def _create_batches(self, audio, labels):
        q, _ = features.extract_melody_cqt(audio['data'], audio['sr'])
        q = np.transpose(q, [1, 0])  # Swap the temporal and bin axes. [temporal, bins]

        cache_idx = random.randint(0, 9999999)  # Lets rely on randomness 🤷 to pock a unique cache spot
        self._cached_features[cache_idx] = q
        position_to_frame = q.shape[0] / len(audio['data'])
        half_window = self._window_size / 2

        def create_sample(arr, centre, label):
            frame_start = int(centre - half_window)
            frame_end = int(centre + half_window)
            if frame_start < half_window or frame_end > q.shape[0] - half_window: return
            arr.append((cache_idx, frame_start, frame_end, label))

        labels = [float(label.split(' ')[0]) for label in labels]  # TODO: here we can add some constraints

        # Create positives
        for label in labels:
            position = int(label * audio['sr'] * position_to_frame)
            create_sample(self.batches, position, 1)

        # Create negatives (just pick some pos i between)
        for i in range(len(labels) - 1):
            position_start = int(labels[i] * audio['sr'] * position_to_frame)
            position_end = int(labels[i + 1] * audio['sr'] * position_to_frame)
            if (position_end - position_start) <= self._label_subdivisions: continue

            step_size = (position_end - position_start) / self._label_subdivisions
            for j in range(1, self._label_subdivisions):
                create_sample(self.batches, position_start + step_size * j, 0)

    def _post_process(self, batch):
        i = self._cached_features[batch[0]][batch[1]:batch[2], :]
        l = np.array([1, 0]) if batch[3] == 1 else np.array([0, 1])
        return i, l

# batcher = MelodyBatcher(['../../../data/processed/default/train'], buffer_size=10000)
# for i in range(10):
#     test = batcher.get_batches()
