import pickle
import random
import numpy as np
import os

import torch

from torch.utils.data import DataLoader, Subset

from PIL import Image
from tqdm import tqdm

from ..feature_extractor import get_features
from .get_obverter_metadata import get_obverter_metadata
from .ObverterDataset import ObverterDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

TRAIN_DATASET_SIZE = 20000
VALID_DATASET_SIZE = 5000
TEST_DATASET_SIZE = 5000

N_DATA_SAMPLES = 100

colors = {
    "red": [1, 0, 0],
    "blue": [0, 0, 1],
    "green": [0, 1, 0],
    "white": [1] * 3,
    "gray": [0.5] * 3,
    "yellow": [1, 1, 0.1],
    "cyan": [0, 1, 1],
    "magenta": [1, 0, 1],
}

object_types = ["box", "sphere", "cylinder", "torus", "ellipsoid"]


def get_obverter_features(dataset="valid"):
    """
    Returns numpy array with matching features
    Args:
        dataset (str) in {'train', 'valid', 'test'}
    """
    dataset_path = "{}/{}_dataset.npy".format(dir_path, dataset)
    features_path = "{}/{}_features.npy".format(dir_path, N_DATA_SAMPLES)

    if not os.path.isfile(features_path):
        images_dict = load_images_dict()
        features = get_features("obverter", images_dict["images"])
        np.save(features_path, features)

    features = np.load(features_path)
    dataset = np.load(dataset_path)

    return features[dataset[0]]


def load_images_dict():
    asset_dir = dir_path + "/assets"
    cache_filename = asset_dir + "/{}_cache.pkl".format(N_DATA_SAMPLES)

    try:
        images_cache = pickle.load(open(cache_filename, "rb"))
        return images_cache
    except FileNotFoundError:
        print("No cache file, trying to create one...")
    except Exception as e:
        print("Error loading cache file", e)
        exit()

    labels = []
    images = []
    labels_to_index = {}
    j = 0
    for color in tqdm(colors):
        for object_type in object_types:
            for i in range(0, N_DATA_SAMPLES):
                path = "{}/{}-{}-{}.png".format(asset_dir, color, object_type, i)
                labels.append((color, object_type, i))
                labels_to_index[(color, object_type, i)] = j
                j += 1
                images.append(
                    np.array(list(Image.open(path).getdata())).reshape((128, 128, 3))
                )

    images = np.stack(images).astype(np.float32) / 255.0
    images_cache = {
        "labels": labels,
        "images": images,
        "labels_to_index": labels_to_index,
    }
    pickle.dump(images_cache, open(cache_filename, "wb"))
    print("Saved cache file {}".format(cache_filename))

    return images_cache


def pick_random_color(exclude=None):
    available_colors = list(colors.keys())
    if exclude is not None:
        available_colors.remove(exclude)

    return random.choice(available_colors)


def pick_random_object_type(exclude=None):
    available_object_types = list(object_types)
    if exclude is not None:
        available_object_types.remove(exclude)

    return random.choice(available_object_types)


def generate_dataset(images_cache, dataset_length=1000):

    n_same = int(0.25 * dataset_length)
    n_same_shape = int(0.3 * dataset_length)
    n_same_color = int(0.2 * dataset_length)
    n_random = dataset_length - n_same_shape - n_same_color - n_same

    pairs = []
    for i in range(n_same):
        object_type, color = pick_random_object_type(), pick_random_color()
        pairs.append(((object_type, color), (object_type, color)))

    for i in range(n_same_shape):
        object_type, color = pick_random_object_type(), pick_random_color()
        color2 = pick_random_color(exclude=color)
        pairs.append(((object_type, color), (object_type, color2)))

    for i in range(n_same_color):
        object_type, color = pick_random_object_type(), pick_random_color()
        object_type2 = pick_random_object_type(exclude=object_type)
        pairs.append(((object_type, color), (object_type2, color)))

    for i in range(n_random):
        object_type, color = pick_random_object_type(), pick_random_color()
        object_type2, color2 = pick_random_object_type(), pick_random_color()
        pairs.append(((object_type, color), (object_type2, color2)))

    input1 = []
    input2 = []
    labels = []
    descriptions = []

    for pair in pairs:
        max_i = N_DATA_SAMPLES
        (object_type1, color1), (object_type2, color2) = pair
        label = object_type1 == object_type2 and color1 == color2

        id1 = random.randint(0, max_i - 1)
        index1 = images_cache["labels_to_index"][color1, object_type1, id1]

        if label:
            available_ids = list(range(id1)) + list(range(id1 + 1, max_i))
            id2 = random.choice(available_ids)
        else:
            id2 = random.randint(0, max_i - 1)
        index2 = images_cache["labels_to_index"][color2, object_type2, id2]

        input1.append(index1)
        input2.append(index2)
        labels.append(int(label))
        descriptions.append(((object_type1, color1), (object_type2, color2)))

    dataset = np.array([input1, input2, labels]).astype(np.int64)
    return dataset, descriptions


def get_obverter_dataset(
    dataset_type="features",
    dataset_length=1000,
    dataset_name="train",
    mean=None,
    std=None,
):
    """
    Args: 
        dataset_type: type of data to be returned by dataset object
                      pick from {"features", "raw", "meta"}
        dataset_length: number of examples in dataset
    """
    images_dict = load_images_dict()

    dataset_path = "{}/{}_dataset.npy".format(dir_path, dataset_name)
    metadata_path = "{}/{}_metadata.npy".format(dir_path, dataset_name)
    if os.path.isfile(dataset_path) and os.path.isfile(metadata_path):
        dataset = np.load(dataset_path)
        metadata = pickle.load(open(metadata_path, "rb"))
    else:
        print("Generating dataset and metadata")
        dataset, metadata = generate_dataset(images_dict, dataset_length=dataset_length)
        np.save(dataset_path, dataset)
        pickle.dump(metadata, open(metadata_path, "wb"))

    # return dataset with precomputed features
    if dataset_type == "features":

        features_path = "{}/{}_features.npy".format(dir_path, N_DATA_SAMPLES)
        if os.path.isfile(features_path):
            features = np.load(features_path)
        else:
            features = get_features("obverter", images_dict["images"])
            np.save(features_path, features)
        assert len(features) == len(images_dict["images"])
        return ObverterDataset(dataset, features)

    # return dataset with raw images
    elif dataset_type == "raw":
        return ObverterDataset(
            dataset, images_dict["images"], mean=mean, std=std, raw=True
        )

    elif dataset_type == "meta":
        encoded_meta = get_obverter_metadata(dataset=dataset_name)
        return ObverterDataset(dataset, encoded_meta.astype(np.float32), metadata=True)

    else:
        raise NotImplementedError()


def get_obverter_dataloader(
    dataset_type="meta", debug=False, batch_size=64, dataset="all"
):

    train_dataset = get_obverter_dataset(
        dataset_type=dataset_type,
        dataset_length=TRAIN_DATASET_SIZE,
        dataset_name="train",
    )

    valid_dataset = get_obverter_dataset(
        dataset_type=dataset_type,
        dataset_length=VALID_DATASET_SIZE,
        dataset_name="valid",
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    test_dataset = get_obverter_dataset(
        dataset_type=dataset_type,
        dataset_length=TEST_DATASET_SIZE,
        dataset_name="test",
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    if debug:  # reduced dataset sizes if debugging
        train_dataset = Subset(train_dataset, range(int(TRAIN_DATASET_SIZE * 0.1)))
        valid_dataset = Subset(valid_dataset, range(int(VALID_DATASET_SIZE * 0.1)))
        test_dataset = Subset(test_dataset, range(int(TEST_DATASET_SIZE * 0.1)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if dataset == "train":
        return train_loader
    if dataset == "valid":
        return valid_loader
    if dataset == "test":
        return test_loader
    else:
        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    get_obverter_dataloader()
