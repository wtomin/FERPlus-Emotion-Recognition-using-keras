from concurrent.futures import ThreadPoolExecutor
from keras.utils import Sequence
import numpy as np
from .dataset import Dataset

def default_collate_fn(samples):
    # support two sample type: list of np.ndarray, dictionary
    n_samples = len(samples)
    assert n_samples>0, 'batch size should larger than 0'
    sample_type = type(samples[0])
    if sample_type == list or sample_type == tuple:
        outputs = []
        for item_id in range(len(samples[0])):
            outputs.append(np.array([samples[sample_id][item_id] for sample_id in range(n_samples)]))
    # elif sample_type == dict:
    #     outputs = {}
    #     for item_key in samples[0].keys():
    #         outputs[item_key] = np.array([samples[sample_id][item_key] for sample_id in range(n_samples)])
    else:
        raise ValueError("current collate_fn only supports dict and np.ndarray!")
    return tuple(outputs)

def default_sampler(dataset):
    return np.arange(0, len(dataset))

class DataLoader(Sequence):

    def __init__(self,
                 dataset: Dataset,
                 batch_size,
                 collate_fn=default_collate_fn,
                 sampler=None,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 ):
        """
 
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None
        :param dataset (Dataset): Data set to load
        :param batch_size (int): how many samples in one batch
        :param shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``True``).
        :param num_workers (int, optional): how many threads to use for data
            loading in one batch. 0 means that the data will be loaded in the main process.
            (default: ``0``)
        :param replacement (bool): samples are drawn with replacement if ``True``, default=False
        :param collate_fn (callable, optional):
        """
        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.indices = []
        self.collate_fn = collate_fn
        self.on_epoch_end()

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        if (index+1)*self.batch_size>len(self.indices):
            assert not self.drop_last, "should drop last!"
            indices = self.indices[index * self.batch_size: ]

        samples = []
        if self.num_workers == 0:
            for i in indices:
                data = self.dataset[i]
                samples.append(data)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for sample in executor.map(lambda i: self.dataset[i], indices):
                    samples.append(sample)
        batchdata = self.collate_fn(samples)
        return batchdata

    def on_epoch_end(self):
        n = len(self.dataset)
        seq = default_sampler(self.dataset)
        if self.sampler is not None:
            self.indices = self.sampler(self.dataset)
        elif self.shuffle:
            np.random.shuffle(seq)
            self.indices = seq
        else:
            self.indices = seq

    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size)) if self.drop_last else int(np.ceil(len(self.dataset) / self.batch_size))

