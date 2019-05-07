import numpy as np
import os

import random
import scipy
import scipy.spatial

dir_path = os.path.dirname(os.path.realpath(__file__))

from .get_shapes_metadata import get_shapes_metadata

DATASET_SIZES = {"train": 20000, "test": 5000, "valid": 5000}


def get_obverter_setup(dataset="train"):

    d_name = "{}/balanced/{}.obverter.npy".format(dir_path, dataset)

    if os.path.isfile(d_name):
        dataset = np.load(d_name)
        return dataset

    metadata = get_shapes_metadata(dataset=dataset)

    dataset_length = DATASET_SIZES[dataset]
    n_same = int(0.3 * dataset_length)
    n_1_edit = int(0.3 * dataset_length)
    n_random = dataset_length - n_same - n_1_edit

    pairs = []
    # generate same example
    for i in range(n_same):
        first_pic_index = np.random.choice(np.arange(metadata.shape[0]))
        first_pic = metadata[first_pic_index]
        same_pics = (np.count_nonzero(metadata != first_pic, axis=1) == 0).nonzero()[0]
        second_pic_index = np.random.choice(same_pics)
        pairs.append([first_pic_index, second_pic_index, 1.0])

    # generate examples which are one edit distance away
    for i in range(n_1_edit):
        first_pic_index = np.random.choice(np.arange(metadata.shape[0]))
        first_pic = metadata[first_pic_index]
        same_pics = (np.count_nonzero(metadata != first_pic, axis=1) == 2).nonzero()[0]
        second_pic_index = np.random.choice(same_pics)
        pairs.append([first_pic_index, second_pic_index, 0.0])

    # random selection
    for i in range(n_random):
        first_pic_index, second_pic_index = np.random.choice(
            np.arange(metadata.shape[0]), 2, replace=False
        )
        label = (
            np.count_nonzero(metadata[first_pic_index] != metadata[second_pic_index])
            == 0
        )
        pairs.append([first_pic_index, second_pic_index, label])

    pairs = np.array(pairs).astype(np.int)
    np.save(d_name, pairs)

    return pairs
