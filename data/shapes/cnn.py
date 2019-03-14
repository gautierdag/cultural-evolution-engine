import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import os

use_gpu = torch.cuda.is_available()


class ShapesDataset(data.Dataset):
    def __init__(self, images):
        super().__init__()

        self.data = images
        self.n_tuples = 0 if len(images.shape) < 5 else images.shape[1]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((250, 250), Image.LINEAR),
            torchvision.transforms.ToTensor(),
            # Needed for pretrained models
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.n_tuples == 0:
            image = self.data[index, :, :, :]
            image = self.transforms(image)
        else:
            imgs = []
            for i in range(self.n_tuples):
                img = self.data[index, i, :, :, :]
                img = self.transforms(img)

                imgs.append(img)
            image = torch.stack(imgs)  # 2 x 3 x 250 x 250
        return image

    def __len__(self):
        return self.data.shape[0]


def cnn_fwd(model, x):
    y = model.features(x)
    y = y.view(y.size(0), -1)
    y = model.classifier[:5](y)
    y = y.detach()
    if use_gpu:
        y = y.cpu()
    return y.numpy()


def get_features(model, dataloader, output_data_folder, file_id):
    for i, x in enumerate(dataloader):
        if use_gpu:
            x = x.cuda()
        if i == 0:
            if len(x.shape) == 5:
                n_tuples = x.shape[1]
            else:
                n_tuples = 0
        if n_tuples == 0:
            y = cnn_fwd(model, x)
        else:
            ys = []
            for j in range(n_tuples):
                x_t = x[:, j, :, :, :]
                y_t = cnn_fwd(model, x_t)
                ys.append(y_t)

            # Here we need to combine 1st elem with 1st elem, etc in the batch
            y = np.asarray(ys)
            y = np.moveaxis(y, 0, 1)
        np.save('{}/{}_{}_features.npy'.format(output_data_folder, file_id, i), y)
        if not use_gpu and i == 5:
            break


def stitch_files(output_data_folder, file_id):
    file_names = ['{}/{}'.format(output_data_folder, f)
                  for f in os.listdir(output_data_folder) if file_id in f]
    file_names.sort(key=os.path.getctime)

    for i, f in enumerate(file_names):
        arr = np.load(f)
        if i == 0:
            features = arr
        else:
            features = np.concatenate((features, arr))

    np.save('{}/{}_features.npy'.format(output_data_folder, file_id), features)


batch_size = 128 if use_gpu else 4

folder = 'different_targets'
train_images = np.load('{}/train.large.input.npy'.format(folder))
val_images = np.load('{}/val.input.npy'.format(folder))
test_images = np.load('{}/test.input.npy'.format(folder))

train_dataset = ShapesDataset(train_images)
val_dataset = ShapesDataset(val_images)
test_dataset = ShapesDataset(test_images)

train_dataloader = DataLoader(
    train_dataset, num_workers=8, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size)
test_dataloader = DataLoader(
    test_dataset, num_workers=8, batch_size=batch_size)

vgg16 = models.vgg16(pretrained=True)
if use_gpu:
    vgg16 = vgg16.cuda()

# print(vgg16)
# print(vgg16.classifier[:5])

vgg16.eval()
for name, param in vgg16.named_parameters():
    if param.requires_grad:
        param.requires_grad = False

output_data_folder = '../data/shapes/{}'.format(folder)

if not os.path.exists(output_data_folder):
    os.mkdir(output_data_folder)

train_features = get_features(
    vgg16, train_dataloader, output_data_folder, 'train')
valid_features = get_features(
    vgg16, val_dataloader, output_data_folder, 'valid')
test_features = get_features(
    vgg16, test_dataloader, output_data_folder, 'test')


# Stitch into one file
stitch_files(output_data_folder, 'train')
stitch_files(output_data_folder, 'valid')
stitch_files(output_data_folder, 'test')
