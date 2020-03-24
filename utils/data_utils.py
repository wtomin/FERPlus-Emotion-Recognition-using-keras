from albumentations.augmentations.transforms import (
    Lambda, Resize, RandomResizedCrop, RandomBrightnessContrast, HorizontalFlip)
from albumentations.core.composition import Compose
import matplotlib.pyplot as plt
import numpy as np
import functools
from keras import backend as K
# keras_vggface has its preprocess_input functionm
# reference: https://github.com/rcmalli/keras-vggface/blob/6943be9d81396ab4083ecff4a8014e99d59b502c/keras_vggface/utils.py#L31
def preprocess_input(x, data_format=None, version=2, **kwargs):
    x_temp = np.copy(x).astype(np.float)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

def aug_preprocess(resize):
    return Compose([
            RandomResizedCrop(height=resize, width=resize,
                               scale=(0.8, 1.0), ratio=(4/5, 5/4)),
            RandomBrightnessContrast(0.1, 0.1),
            HorizontalFlip(p=0.5),
            Lambda(image=preprocess_input)
        ], p=1) # always apply augmentation

def preprocess(resize):
    return Compose([
            Resize(height=resize, width=resize, p=1),
            Lambda(image=preprocess_input)
        ], p=1) # always apply 

def plotImages(images_arr, n_images_show=5):
    if len(images_arr)>n_images_show:
        print("Only show the first {} images".format(n_images_show))
    images_arr = images_arr[:n_images_show, ...]
    images_arr +=  np.array([131.0912, 103.8827, 91.4953])
    images_arr = (images_arr - np.min(images_arr))/np.max(images_arr)
    fig, axes = plt.subplots(1, len(images_arr), figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

