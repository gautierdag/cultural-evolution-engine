import numpy as np
import matplotlib.pyplot as plt
import sys


def display_image(img):
    plt.clf()
    img = np.transpose(img, (2, 0, 1))
    b, g, r = img[0, ...], img[1, ...], img[2, ...]
    rgb = np.asarray([r, g, b])
    transp = np.transpose(rgb, (1, 2, 0))
    plt.imshow(transp)
    plt.show()


if __name__ == "__main__":

    folder = "different_targets"
    npy_file = "{}/train.large.input.npy".format(folder)

    assert len(sys.argv) > 1, "Indicate an index!"

    img = np.load(npy_file)[int(sys.argv[1])]
    print(img.shape)

    display_image(img)
