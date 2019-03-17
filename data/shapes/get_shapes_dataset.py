import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler
from generate_shapes import generate_shapes_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_dataloaders(batch_size=16, k=3):
    """
    Returns dataloader for the train/valid/test datasets
    Args:
        batch_size: batch size to be used in the dataloader
        k: number of distractors to be used in training
    """
    train_features = np.load(dir_path + "/train_features.npy")
    valid_features = np.load(dir_path + "/valid_features.npy")
    test_features = np.load(dir_path + "/test_features.npy")

    n_image_features = valid_features.shape[1]

    train_dataset = ImageDataset(train_features)

    # All features are normalized with train mean and std
    valid_dataset = ImageDataset(
        valid_features, mean=train_dataset.mean, std=train_dataset.std
    )
    test_dataset = ImageDataset(
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

    return n_image_features, train_data, valid_data, test_data


def get_shapes_dataset(batch_size=16, k=3):
    if not os.path.exists(dir_path + "/train_features.npy"):
        print("Features files not present - generating dataset")
        generate_shapes_dataset()

    return get_dataloaders(batch_size=batch_size, k=k)


get_shapes_dataset()
