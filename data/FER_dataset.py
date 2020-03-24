#from tensorflow.keras.utils import Sequence
#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.preprocessing import image 
from keras.utils import to_categorical
import pandas as pd
from .dataset import Dataset
from utils.data_utils import aug_preprocess, preprocess
from PATH import PATH
import numpy as np
PRESET_VARS = PATH()

class FER_dataset(Dataset):
    def __init__(self, opt, train_mode='Train', transform = None):
        super(FER_dataset, self).__init__(opt, train_mode, transform)
        self._name = 'FER_dataset'
        self._opt = opt
        self.categories = PRESET_VARS.FER2013.categories
        self.n_classes = len(self.categories)
        self._train_mode = train_mode
        self._data = self._read_pixel_emotion()
        self._read_dataset_paths()
        if transform is not None:
            self._transform = transform
        else:
            self._transform = self.create_transform()
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        image = None
        label = None
        label = self.data.iloc[index]['emotion']
        pixels = self.data.iloc[index]['pixels']
        pixels = [int(x) for x in pixels.strip().split(' ')]
        img_array = np.array(pixels).reshape((48, 48, 1)).repeat(3, axis=2).astype(np.uint8)
        image = self._transform(image=img_array)['image']
        # # pack data
        # sample = {'image': image,
        #           'label': to_categorical(label, num_classes = self.n_classes),
        #           'index': index
        #           }
        return image, to_categorical(label, num_classes = self.n_classes)
    def _read_pixel_emotion(self):
        FER2013_csv = PRESET_VARS.FER2013.data_file
        df = pd.read_csv(FER2013_csv)
        return df
    def _read_dataset_paths(self):
        # renew the index
        if self._train_mode == 'Train':
            self.data = self._data[self._data['Usage']=='Training'].reset_index()
        elif self._train_mode == 'Validation':
            self.data  = self._data[self._data['Usage']=='PublicTest'].reset_index()
        elif self._train_mode == 'Test':
            self.data  = self._data[self._data['Usage']=='PrivateTest'].reset_index()
        else:
            raise ValueError("{} is not correct!".format(self.train_mode))
        self._ids = np.arange(len(self.data)) 
        self._dataset_size = len(self._ids)

    def __len__(self):
        return self._dataset_size
    def create_transform(self):
        if self._train_mode == 'Train':
            return aug_preprocess(resize = self._opt.image_size)
        else:
            return preprocess(resize = self._opt.image_size)

class FERPlus_dataset(FER_dataset):
    def __init__(self, *args, **kwargs):
        super(FERPlus_dataset, self).__init__(*args, **kwargs)
        self._name = 'FERPlus_dataset'
        self.categories = PRESET_VARS.FERPlus.categories
        self.n_classes = len(self.categories)
        self.update_labels()
        self._read_dataset_paths()

    def update_labels(self):
        """Update dataset to use FerPlus labels, rather Fer2013 dataset labels

        Aim to reproduce the Microsoft CNTK cleaning process. These are based
        on some heuristics about the level of ambiguity in the annotator labels
        that should be tolerated to ensure that the dataset is moderately
        clearn. We generate hard labels, rather than soft ones for evaluation.
        """
        FERplus_csv =  PRESET_VARS.FERPlus.data_file
        df = pd.read_csv(FERplus_csv)
        labels = df[self.categories + ['unknown', 'NF']].values
        # following CNTK processing - there are three reasons to drop examples:
        # (1) If the majority votes for either "unknown-face" or "not-face"
        # (2) If more than three votes share the maximum voting value
        # (3) If the max votes do not account for more than half of the votes
        num_votes = np.sum(labels, 1)
        to_drop = np.zeros((labels.shape[0], 1))
        for ii in range(labels.shape[0]):
            max_vote = np.max(labels[ii,:])
            max_vote_emos = np.where(labels[ii,:] == max_vote)[0]
            drop = any([x in [8, 9] for x in max_vote_emos]) # remove unknown and NF
            num_max_votes = max_vote_emos.size
            drop = drop or num_max_votes >= 3
            drop = drop or (num_max_votes * max_vote < 0.5 * num_votes[ii])
            to_drop[ii] = drop
        assert to_drop.sum() == 3154, 'unexpected number of dropped votes'
        drop_ids = [i for i, x in enumerate(to_drop) if x ]
        df = df.drop(drop_ids, axis=0)
        self._data = self._data.drop(drop_ids, axis=0)
        updated_labels = np.argmax(df[self.categories].values, 1)
        self._data.loc[:,'emotion'] = updated_labels

