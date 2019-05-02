import numpy as np
import torchvision.transforms


class ObverterDataset:
    def __init__(
        self, dataset, features, mean=None, std=None, metadata=False, raw=False
    ):
        self.metadata = metadata
        self.raw = raw
        self.dataset = dataset

        # encoded metadata (using a vocabulary)
        self.features = features

        # else raw image or pre computed image features
        if mean is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero

        self.mean = mean
        self.std = std

        if not raw and not metadata:
            self.features = (features - self.mean) / (2 * self.std)

        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def __getitem__(self, idx):
        if self.metadata:
            first_image = self.features[idx][0]
            second_image = self.features[idx][1]
        else:
            first_image = self.features[self.dataset[0][idx]]
            second_image = self.features[self.dataset[1][idx]]
            if self.raw:
                first_image = self.transforms(first_image)
                second_image = self.transforms(second_image)

        target = self.dataset[2][idx]  # label
        return (first_image, second_image, target)

    def __len__(self):
        return self.dataset.shape[1]
