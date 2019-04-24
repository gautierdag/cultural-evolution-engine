import pickle
import os
import numpy as np
from .encode_metadata import encode_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def get_obverter_metadata(dataset="valid", first_picture_only=False):
    """
    Args:
        dataset (str, opt) from {"train", "valid", "test"}
    returns one hot encoding of metada - compressed version of true concepts
    """
    meta = np.load("{}/{}_metadata.npy".format(dir_path, dataset))

    # compressed_test_images = np.zeros((len(test_meta), 2))
    compressed_test_images = encode_metadata(meta)

    if first_picture_only:
        compressed_test_images = compressed_test_images[
            :, 0, :
        ]  # keep only first image
        one_hot_derivations = one_hot(compressed_test_images).sum(axis=1)
    else:
        one_hot_derivations = one_hot(compressed_test_images).sum(axis=2)

    return one_hot_derivations


if __name__ == "__main__":
    get_obverter_metadata()
