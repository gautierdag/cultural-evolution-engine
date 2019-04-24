import pickle
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def get_shapes_metadata(dataset="test", one_hot_encode=True):
    """
    Args:
        dataset (str, opt) from {"train", "valid", "test"}
    returns one hot encoding of metada - compressed version of true concepts
    @TODO implement loading from file rather than loading each time
    """

    test_meta = pickle.load(
        open(dir_path + "/balanced/{}.metadata.p".format(dataset), "rb")
    )

    compressed_test_images = np.zeros((len(test_meta), 5))
    for i, m in enumerate(test_meta):
        pos_h, pos_w = (np.array(m["shapes"]) != None).nonzero()
        pos_h, pos_w = pos_h[0], pos_w[0]
        color = m["colors"][pos_h][pos_w]
        shape = m["shapes"][pos_h][pos_w]
        size = m["sizes"][pos_h][pos_w]
        compressed_test_images[i] = np.array([color, shape, size, pos_h, pos_w])
    compressed_test_images = compressed_test_images.astype(np.int)

    # return one hot encoding
    # note this will have 15 dimensions and not 14 as expected
    if one_hot_encode:
        one_hot_derivations = one_hot(compressed_test_images).reshape(
            compressed_test_images.shape[0], -1
        )
        return one_hot_derivations

    # return encoding based on vocabulary
    else:
        # need to hard code because we only have access to integers..
        true_label_dist = [3, 3, 2, 3, 3]
        for i in range(1, 5):
            compressed_test_images[:, i] += true_label_dist[i - 1]
            true_label_dist[i] += true_label_dist[i - 1]

        return compressed_test_images


if __name__ == "__main__":
    m = get_shapes_metadata(one_hot_encode=False)
    print(m.shape)
