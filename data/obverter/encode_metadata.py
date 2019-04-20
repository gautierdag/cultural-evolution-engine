import pickle
import numpy as np
import os
from collections import defaultdict

dir_path = os.path.dirname(os.path.realpath(__file__))


def unk_index():
    return 0


class Vocab(object):
    """
    Vocab object to create vocabulary and load if exists
    """

    def __init__(self, vocab_set, name="object"):
        self.vocab_size = len(vocab_set)
        self.file_path = dir_path + "/{}_{}_vocab.pckl".format(self.vocab_size, name)
        self.build_vocab(vocab_set)

    def save_vocab(self):
        with open(self.file_path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def build_vocab(self, vocab_set):
        self.stoi = defaultdict(unk_index)
        self.stoi["<UNK>"] = 0
        self.itos = ["<UNK>"]
        for i, token in enumerate(vocab_set):
            self.itos.append(token)
            self.stoi[token] = i + 1

        self.save_vocab()


def get_all_possible_colors_objects(metadata):
    colors = set()
    objects = set()
    for ((obj1, col1), (obj2, col2)) in metadata:
        objects.add(obj1)
        colors.add(col1)
        objects.add(obj2)
        colors.add(col2)
    return list(colors), list(objects)


def encode_metadata(metadata, colors_vocab=None, object_vocab=None, combined=False):
    if colors_vocab is None and object_vocab is None:
        colors, objects = get_all_possible_colors_objects(metadata)
        if combined:
            colors_vocab = Vocab(colors + objects, name="combined")
            object_vocab = colors_vocab
        else:
            colors_vocab = Vocab(colors, name="color")
            object_vocab = Vocab(objects, name="object")

    encoded_metadata = []
    for ((obj1, col1), (obj2, col2)) in metadata:
        encoded_metadata.append(
            [
                [object_vocab.stoi[obj1], colors_vocab.stoi[col1]],
                [object_vocab.stoi[obj2], colors_vocab.stoi[col2]],
            ]
        )
    return np.array(encoded_metadata), colors_vocab, object_vocab


if __name__ == "__main__":
    tmp = [
        (("cylinder", "red"), ("cylinder", "red")),
        (("cylinder", "red"), ("sphere", "red")),
        (("sphere", "red"), ("sphere", "red")),
        (("cylinder", "blue"), ("cylinder", "blue")),
    ]

    colors = {
        "red": [1, 0, 0],
        "blue": [0, 0, 1],
        "green": [0, 1, 0],
        "white": [1] * 3,
        "gray": [0.5] * 3,
        "yellow": [1, 1, 0.1],
        "cyan": [0, 1, 1],
        "magenta": [1, 0, 1],
    }

    object_types = ["box", "sphere", "cylinder", "torus", "ellipsoid"]

    a, colors_vocab, object_vocab = encode_metadata(tmp)
    print(a)
    a, colors_vocab, object_vocab = encode_metadata(tmp, combined=True)
    print(a)
    print(a.shape)
