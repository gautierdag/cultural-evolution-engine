import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms
import torch.utils.data as data
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16  # batch size used to extract features


class ImageDataset(data.Dataset):
    def __init__(self, images):
        super().__init__()

        self.data = images
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # Normalize to (-1, 1)
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        image = self.data[index, :, :, :]
        image = self.transforms(image)
        return image

    def __len__(self):
        return self.data.shape[0]


def get_features(model, images):
    print("Extracting features")

    dataloader = DataLoader(ImageDataset(images), batch_size=BATCH_SIZE)

    features = []
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device)
        y = model.features(x)
        y = y.view(y.size(0), -1).detach().cpu().numpy()
        features.append(y)

    # concatenate all features
    features = np.concatenate(features, axis=0)
    return features
