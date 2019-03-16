import os
import pickle
import numpy as np

import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from generate_dataset import *
from cnn import ShapesDataset, get_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 16  # batch size used to extract features

if __name__ == "__main__":

    folder_name = "balanced"
    np.random.seed(SEED)

    # From Serhii's original experiment
    train_size = 74504
    val_size = 8279
    test_size = 40504

    # --- Generate Datasets ----
    train_data, valid_data, test_data = get_datasets(
        train_size, val_size, test_size, get_dataset_balanced, SEED
    )

    sets = {"train": train_data, "valid": valid_data, "test": test_data}

    # --- Save Generated Datasets ----
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = os.path.join(dir_path, folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for set_name, set_data in sets.items():
        set_inputs = np.asarray([image.data[:, :, 0:3] for image in set_data])
        np.save("{}/{}.input".format(folder_name, set_name), set_inputs)
        set_metadata = [image.metadata for image in set_data]
        pickle.dump(
            set_metadata, open("{}/{}.metadata.p".format(folder_name, set_name), "wb")
        )

    # --- Getting Features for the Generated Images ----

    # Load Pretrained model and move model to device
    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device)
    vgg16.eval()

    # get features from train, valid, and test
    for set_name in sets.keys():
        images = np.load("{}/{}.input.npy".format(folder_name, set_name))
        shapes_dl = DataLoader(ShapesDataset(images), batch_size=BATCH_SIZE)
        features = get_features(vgg16, shapes_dl)
        np.save("{}/{}_features.npy".format(dir_path, set_name), features)
