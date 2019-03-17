import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import Model
from train_utils import train_one_epoch, evaluate
from data.shapes import get_shapes_dataset, ShapesVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

debugging = False

SEED = 42


def seed_torch(seed=42):
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


seed_torch(seed=SEED)

if __name__ == "__main__":

    EPOCHS = 1000 if not debugging else 10  # 2
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    BATCH_SIZE = 128 if not debugging else 4
    MAX_SENTENCE_LENGTH = 13 if not debugging else 5
    K = 3  # number of distractors
    vocab_size = 10

    # Load Vocab
    vocab = ShapesVocab(vocab_size)

    # Load data
    n_image_features, train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=BATCH_SIZE, k=K
    )

    model_id = "baseline"

    ################# Print info ####################
    print("----------------------------------------")
    print("Model id: {}".format(model_id))
    print("|V|: {}".format(vocab_size))
    print("L: {}".format(MAX_SENTENCE_LENGTH))
    #################################################

    model = Model(n_image_features, vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max' patience=30,
                                  threshold=0.005, threshold_mode='rel')

    losses_meters = []
    eval_losses_meters = []
    accuracy_meters = []
    eval_accuracy_meters = []

    # Train
    for epoch in range(EPOCHS):

        epoch_loss_meter, epoch_acc_meter = train_one_epoch(
            model, train_data, optimizer, vocab.bound_idx, MAX_SENTENCE_LENGTH
        )

        losses_meters.append(epoch_loss_meter)
        accuracy_meters.append(epoch_acc_meter)

        eval_loss_meter, eval_acc_meter, eval_messages = evaluate(
            model, valid_data, vocab.bound_idx, MAX_SENTENCE_LENGTH
        )

        scheduler.step(eval_acc_meter.avg)

        eval_losses_meters.append(eval_loss_meter)
        eval_accuracy_meters.append(eval_acc_meter)

        # Skip for now
        print(
            "Epoch {}, average train loss: {}, average val loss: {}, \
                average accuracy: {}, average val accuracy: {}".format(
                epoch,
                losses_meters[epoch].avg,
                eval_losses_meters[epoch].avg,
                accuracy_meters[epoch].avg,
                eval_accuracy_meters[epoch].avg,
            )
        )

    # Evaluate best model on test data
    _, test_acc_meter, test_messages = evaluate(
        model, test_data, vocab.bound_idx, MAX_SENTENCE_LENGTH
    )
    print("Test accuracy: {}".format(test_acc_meter.avg))
