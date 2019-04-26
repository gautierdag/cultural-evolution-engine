import numpy as np
import os
from tqdm import tqdm

import torch
import torchvision.transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16  # batch size used to extract features


class ImageDataset(data.Dataset):
    def __init__(self, images):
        super().__init__()

        self.data = images

        H, W = images.shape[1], images.shape[2]
        if H != 128 or W != 128:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    # Resize
                    torchvision.transforms.Resize((128, 128), Image.LINEAR),
                    torchvision.transforms.ToTensor(),
                    # Normalize to (-1, 1)
                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
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


def get_features(task, images):
    print("Extracting features")

    model_name = "{}/extractor_{}.p".format(dir_path, task)
    if not os.path.isfile(model_name):
        return ValueError(
            "Feature Extractor for {} missing. Train baseline using 'raw' features.".format(
                task
            )
        )

    model = torch.load(model_name, map_location=lambda storage, location: storage)
    model.to(device)

    dataloader = DataLoader(ImageDataset(images), batch_size=BATCH_SIZE)

    features = []
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device)
        y = model(x)
        y = y.view(y.size(0), -1).detach().cpu().numpy()
        features.append(y)

    # concatenate all features
    features = np.concatenate(features, axis=0)
    return features
