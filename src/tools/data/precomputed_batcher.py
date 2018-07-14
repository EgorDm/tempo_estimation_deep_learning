from collections import MutableSequence
from typing import Tuple
import numpy as np
import glob
import os
import msgpack
import msgpack_numpy as m
from src.tools.data.batcher import Batcher

m.patch()


class PrecomputedBatcher(Batcher):
    def __init__(self, feature, datasets=None, sample_files=None, buffer_size=10, batch_size=20, randomize=True, repeat_dataset=True):
        self.feature = feature
        self.inputs = None
        self.labels = None
        super().__init__(datasets, sample_files, buffer_size, batch_size, randomize, repeat_dataset)

    def _load_dataset_entries(self, dataset):
        if not os.path.exists(dataset): raise Exception(f'Dataset not found {dataset}')
        dataset = dataset.replace('\\', '/')

        files = glob.glob(f'{dataset}/*{self.feature}.dat')
        return ['.'.join(file.replace('\\', '/').split('.')[:-1]) for file in files]

    def _create_samples_from_entry(self, entry):
        with open(f'{entry}.dat', 'rb') as f:
            inputs, labels = msgpack.unpack(f)

        if self.inputs is None:
            self.inputs = inputs
            self.labels = labels
        else:
            self.inputs = np.concatenate((self.inputs, inputs), axis=0)
            self.labels = np.concatenate((self.labels, labels), axis=0)

    def sample_count(self):
        return self.inputs.shape[0] if self.inputs is not None else 0

    def get_batch(self, count=None) -> Tuple[np.ndarray, np.ndarray]:
        if count is None: count = self.batch_size
        if self.sample_count() < count: self._buffer_batches()  # Buffer batches if there are not enough

        count = min(self.sample_count(), count)
        if count == 0: return None

        inputs = self.inputs[-count:]
        labels = self.labels[-count:]

        self.inputs = np.delete(self.inputs, slice(-count, None), axis=0)
        self.labels = np.delete(self.labels, slice(-count, None), axis=0)

        return inputs, labels

#
# batcher = PrecomputedBatcher('melody', '../../../data/processed/default/train')
# test = batcher.get_batch(6000)