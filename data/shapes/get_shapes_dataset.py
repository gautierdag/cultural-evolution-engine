import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from .ShapesDataset import ShapesDataset, ImagesSampler
from .generate_shapes import generate_shapes_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_dataloaders(batch_size=16, k=3, debug=False, dataset="all"):
    """
    Returns dataloader for the train/valid/test datasets
    Args:
        batch_size: batch size to be used in the dataloader
        k: number of distractors to be used in training
        debug (bool, optional): whether to use a much smaller subset of train data
    """
    train_features = np.load(dir_path + "/train_features.npy")
    valid_features = np.load(dir_path + "/valid_features.npy")
    test_features = np.load(dir_path + "/test_features.npy")

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


def get_shapes_dataset(batch_size=16, k=3, debug=False, dataset="all"):
    """
    Args:
        batch_size (int, opt): batch size of dataloaders
        k (int, opt): number of distractors
    """
    if not os.path.exists(dir_path + "/train_features.npy"):
        print("Features files not present - generating dataset")
        generate_shapes_dataset()

    return get_dataloaders(batch_size=batch_size, k=k, debug=debug, dataset=dataset)


def get_shapes_features(dataset="test"):
    """
    Returns numpy array with matching features
    Args:
        dataset (str) in {'train', 'valid', 'test'}
    """
    return np.load(dir_path + "/{}_features.npy".format(dataset))
