import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from .ShapesDataset import ShapesDataset, ImagesSampler
from .generate_shapes import generate_shapes_dataset
from .get_shapes_metadata import get_shapes_metadata
from ..feature_extractor import get_features

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_shapes_features(dataset="test"):
    """
    Returns numpy array with matching features
    Args:
        dataset (str) in {'train', 'valid', 'test'}
    """
    features_path = "{}/{}_features.npy".format(dir_path, dataset)

    if not os.path.isfile(features_path):
        images = np.load("{}/balanced/{}.input.npy".format(dir_path, dataset))
        features = get_features("shapes", images)
        np.save(features_path, features)
        assert len(features) == len(images)

    return np.load(features_path)


def get_dataloaders(
    batch_size=16, k=3, debug=False, dataset="all", dataset_type="features"
):
    """
    Returns dataloader for the train/valid/test datasets
    Args:
        batch_size: batch size to be used in the dataloader
        k: number of distractors to be used in training
        debug (bool, optional): whether to use a much smaller subset of train data
        dataset (str, optional): whether to return a specific dataset or all
                                 options are {"train", "valid", "test", "all"}
                                 default: "all"
        dataset_type (str, optional): what datatype encoding to use: {"meta", "features", "raw"}
                                      default: "features"
    """
    if dataset_type == "raw":
        train_features = np.load(dir_path + "/balanced/train.input.npy")
        valid_features = np.load(dir_path + "/balanced/valid.input.npy")
        test_features = np.load(dir_path + "/balanced/test.input.npy")

        train_dataset = ShapesDataset(train_features, raw=True)

        # All features are normalized with train mean and std
        valid_dataset = ShapesDataset(
            valid_features, mean=train_dataset.mean, std=train_dataset.std, raw=True
        )
        test_dataset = ShapesDataset(
            test_features, mean=train_dataset.mean, std=train_dataset.std, raw=True
        )

    if dataset_type == "features":

        train_features = get_shapes_features(dataset="train")
        valid_features = get_shapes_features(dataset="valid")
        test_features = get_shapes_features(dataset="test")

        if debug:
            train_features = train_features[:10000]

        train_dataset = ShapesDataset(train_features)

        # All features are normalized with train mean and std
        valid_dataset = ShapesDataset(
            valid_features, mean=train_dataset.mean, std=train_dataset.std
        )
        test_dataset = ShapesDataset(
            test_features, mean=train_dataset.mean, std=train_dataset.std
        )

    if dataset_type == "meta":
        train_meta = get_shapes_metadata(dataset="train")
        valid_meta = get_shapes_metadata(dataset="valid")
        test_meta = get_shapes_metadata(dataset="test")

        train_dataset = ShapesDataset(train_meta.astype(np.float32), metadata=True)
        valid_dataset = ShapesDataset(valid_meta.astype(np.float32), metadata=True)
        test_dataset = ShapesDataset(test_meta.astype(np.float32), metadata=True)

    train_data = DataLoader(
        train_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(train_dataset, k, shuffle=True),
            batch_size=batch_size,
            drop_last=True,
        ),
    )

    valid_data = DataLoader(
        valid_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(valid_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=True,
        ),
    )

    test_data = DataLoader(
        test_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(test_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=True,
        ),
    )
    if dataset == "train":
        return train_data
    if dataset == "valid":
        return valid_data
    if dataset == "test":
        return test_data
    else:
        return train_data, valid_data, test_data


def get_shapes_dataloader(
    batch_size=16, k=3, debug=False, dataset="all", dataset_type="features"
):
    """
    Args:
        batch_size (int, opt): batch size of dataloaders
        k (int, opt): number of distractors
    """

    if (
        not os.path.exists(dir_path + "/train_features.npy")
        and dataset_type == "features"
    ):
        print("Features files not present - generating dataset")
        generate_shapes_dataset()

    return get_dataloaders(
        batch_size=batch_size,
        k=k,
        debug=debug,
        dataset=dataset,
        dataset_type=dataset_type,
    )

