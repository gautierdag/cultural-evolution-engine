import pickle
import random
import numpy as np
import os

import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm

from ..feature_extractor import get_features
from .encode_metadata import encode_metadata
from .ObverterDataset import ObverterDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_path = os.path.dirname(os.path.realpath(__file__))

TRAIN_DATASET_SIZE = 10000
VALID_DATASET_SIZE = 5000
DEBUG_TRAIN_DATASET_SIZE = 1000
DEBUG_VALID_DATASET_SIZE = 500
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


def load_images_dict():
    asset_dir = dir_path + "/assets"
    cache_filename = asset_dir + "/{}_cache.pkl".format(N_DATA_SAMPLES)

    print("Looking for cache file {}".format(cache_filename))
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
    dataset_type="features", dataset_length=1000, col_vocab=None, obj_vocab=None
):
    """
    Args: 
        dataset_type: type of data to be returned by dataset object
                      pick from {"features", "raw", "meta", "meta_combined"}
        dataset_length: number of examples in dataset
        col_vocab: color vocab object (used for metadata)
        obj_vocab: object vocab object (used for metadata)
        Note when meta_combined the col and obj vocabs are the same
    """
    images_dict = load_images_dict()

    dataset_path = "{}/dataset_{}.npy".format(dir_path, dataset_length)
    metadata_path = "{}/metadata_{}.npy".format(dir_path, dataset_length)
    if os.path.isfile(dataset_path) and os.path.isfile(metadata_path):
        print("Loading dataset and metadata from file")
        dataset = np.load(dataset_path)
        metadata = pickle.load(open(metadata_path, "rb"))
    else:
        print("Generating dataset and metadata")
        dataset, metadata = generate_dataset(images_dict, dataset_length=dataset_length)
        np.save(dataset_path, dataset)
        pickle.dump(metadata, open(metadata_path, "wb"))

    # return dataset with precomputed features
    if dataset_type == "features":
        # Load Pretrained model and move model to device
        vgg16 = models.vgg16(pretrained=True)
        vgg16.to(device)
        vgg16.eval()

        features_path = "{}/{}_features.npy".format(dir_path, N_DATA_SAMPLES)
        if os.path.isfile(features_path):
            features = np.load(features_path)
        else:
            features = get_features(vgg16, images_dict["images"])
            np.save(features_path, features)
        assert len(features) == len(images_dict["images"])
        return ObverterDataset(dataset, features), col_vocab, obj_vocab

    # return dataset with raw images
    elif dataset_type == "raw":
        return ObverterDataset(dataset, images_dict["images"]), col_vocab, obj_vocab

    elif dataset_type == "meta" or dataset_type == "meta_combined":
        combined = "combined" in dataset_type
        if col_vocab is None and obj_vocab is None:
            encoded_meta, col_vocab, obj_vocab = encode_metadata(
                metadata, combined=combined
            )
        else:
            encoded_meta, _, _ = encode_metadata(
                metadata, colors_vocab=col_vocab, object_vocab=obj_vocab
            )
        return (
            ObverterDataset(dataset, encoded_meta, metadata=True),
            col_vocab,
            obj_vocab,
        )

    else:
        raise NotImplementedError()


def get_obverter_dataloader(dataset_type="meta", debugging=False, batch_size=64):
    length_train = TRAIN_DATASET_SIZE * (0.1 if debugging else 1)
    length_valid = VALID_DATASET_SIZE * (0.1 if debugging else 1)

    train_dataset, col_vocab, obj_vocab = get_obverter_dataset(
        dataset_type=dataset_type, dataset_length=length_train
    )
    valid_dataset, _, _ = get_obverter_dataset(
        dataset_type=dataset_type,
        dataset_length=length_valid,
        col_vocab=col_vocab,  # use vocab from train
        obj_vocab=obj_vocab,  # use vocab from train
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    return train_loader, valid_loader, (col_vocab, obj_vocab)


if __name__ == "__main__":
    get_obverter_dataloader()
