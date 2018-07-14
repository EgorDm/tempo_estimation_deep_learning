import glob
import os
import random
from typing import Tuple, Union, Optional
import numpy as np
from tqdm import tqdm


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

        progress = tqdm(total=self.buffer_size, desc='Filling buffer')
        while self.sample_count() < self.buffer_size:
            self.cursor += 1
            if self.cursor >= len(self.train_data_files):
                if not self.repeat_dataset: return
                self.cursor = 0
                self.epoch += 1

            entry = self.train_data_files[self.cursor]
            self._create_samples_from_entry(entry)

            progress.update(self.sample_count() - progress.n)
        progress.close()

        if self.randomize: random.shuffle(self.samples)

        # We push the leftovers to front since they had their share of suffling and we want to empty cache asap
        self.samples += tmp

    def _create_samples_from_entry(self, entry): pass

    def _post_process_sample(self, sample) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Post process the batch before returning it to the model
        """
        return sample

    def get_batch(self, count=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if count is None: count = self.batch_size
        if self.sample_count() < count: self._buffer_batches()  # Buffer batches if there are not enough

        count = min(self.sample_count(), count)
        if count == 0: return None

        inputs = []
        labels = []
        for samples in self.samples[-count:]:
            i, l = self._post_process_sample(samples)
            inputs.append(i)
            labels.append(l)

        del self.samples[-count:]
        return np.stack(inputs, axis=0), np.stack(labels, axis=0)

    def generator(self):
        while 1:
            x, y = self.get_batch()
            if self.sample_count() == 0: break
            yield x, y

    def sample_count(self):
        return len(self.samples)


class CacheableBatcher(Batcher):
    """
    Requires every batch to specify the cache position as the first element
    """

    def __init__(self, datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True, repeat_dataset=True):
        self._cached_features = {}

        super().__init__(datasets, sample_files, buffer_size, batch_size, randomize, repeat_dataset)

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
