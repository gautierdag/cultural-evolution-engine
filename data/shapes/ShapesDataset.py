import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler


class ShapesDataset:
    def __init__(self, features, mean=None, std=None):
        if mean is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std
        self.features = (features - self.mean) / (2 * self.std)

    def __getitem__(self, indices):
        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.features[d_idx])

        return (self.features[target_idx], distractors)

    def __len__(self):
        return self.features.shape[0]


class ImagesSampler(Sampler):
    def __init__(self, data_source, k, shuffle):
        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        if self.shuffle:
            targets = torch.randperm(self.n).tolist()
        else:
            targets = list(range(self.n))

        for t in targets:
            arr = np.zeros(self.k + 1, dtype=int)  # distractors + target
            arr[0] = t
            distractors = random.sample(range(self.n), self.k)
            while t in distractors:
                distractors = random.sample(range(self.n), self.k)
            arr[1:] = np.array(distractors)

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.n
