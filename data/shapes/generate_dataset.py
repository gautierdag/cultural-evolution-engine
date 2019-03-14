from image_utils import *
import numpy as np
from random import shuffle


def get_datasets(train_size, val_size, test_size, f_get_dataset, seed):
    train_data = f_get_dataset(train_size, seed)
    val_data = f_get_dataset(val_size, seed)
    test_data = f_get_dataset(test_size, seed)
    return train_data, val_data, test_data


def get_dataset_balanced(size, seed):
    np.random.seed(seed)
    images = []
    for i in range(size):
        images.append(get_image(seed+i))
    shuffle(images)
    return images


def get_dataset_unbalanced(size, seed, least_freq_shape=SHAPE_CIRCLE, least_freq_ratio=0.1):
    np.random.seed(seed)
    images = []
    n_unfreq_shapes = least_freq_ratio * size
    for i in range(size):
        if i < n_unfreq_shapes:
            shape = least_freq_shape
        else:
            shape = least_freq_shape + \
                1 if np.random.randint(2) == 0 else least_freq_shape + 2

        images.append(get_image(seed+i, shape))
    shuffle(images)
    return images


def get_dataset_different_targets(size, seed):
    images = []
    for i in range(size):
        np.random.seed(seed+i)
        shape = np.random.randint(N_SHAPES)
        color = np.random.randint(N_COLORS)
        img1 = get_image(seed+i, shape, color)
        # Different size and/or location
        img2 = get_image(seed+(i+1)*2, shape, color)
        j = 0
        while img1.metadata == img2.metadata:
            img2 = get_image(seed+i*2+j+1, shape, color)
            j += 1
        images.append((img1, img2))
    shuffle(images)
    return images
