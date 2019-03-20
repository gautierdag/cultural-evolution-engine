import pickle
import argparse
import os
import sys
import torch

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Sender, Receiver, Trainer
from train_utils import *
from data.shapes import get_shapes_dataset, ShapesVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shapes_trainer(params, sender_file, receiver_file, writer=None):
    """
    Args:
        params (dict, required): params must be a dict that contains
            vocab_size, batch_size, k and epochs
        sender_file (str, required): filename of sender model
        receiver_file (str, required): filename of receiver model
        writer (optional): tensorboard writer
    """
    # Load Vocab
    vocab = ShapesVocab(params.vocab_size)

    # Load data
    n_image_features, train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=params.batch_size, k=params.k
    )

    sender = torch.load(sender_file)
    receiver = torch.load(receiver_file)
    print(sender)
    print(receiver)
    model = Trainer(sender, receiver)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(mode="max")

    # Train
    for epoch in range(params.epochs):
        loss_meter, acc_meter = train_one_epoch(model, train_data, optimizer)
        eval_loss_meter, eval_acc_meter, eval_messages = evaluate(model, valid_data)

        if writer is not None:
            writer.add_scalar("avg_train_epoch_loss", loss_meter.avg, epoch)
            writer.add_scalar("avg_valid_epoch_loss", eval_loss_meter.avg, epoch)
            writer.add_scalar("avg_train_epoch_acc", acc_meter.avg, epoch)
            writer.add_scalar("avg_valid_epoch_acc", eval_acc_meter.avg, epoch)

        early_stopping.step(eval_acc_meter.avg)
        if early_stopping.num_bad_epochs == 0:
            torch.save(model.state_dict(), "{}/best_model".format(run_folder))

        # Skip for now
        print(
            "Epoch {}, average train loss: {}, average val loss: {}, \
                average accuracy: {}, average val accuracy: {}".format(
                epoch,
                loss_meter.avg,
                eval_loss_meter.avg,
                acc_meter.avg,
                eval_acc_meter.avg,
            )
        )

        if early_stopping.is_converged:
            print("Converged in epoch {}".format(epoch))
            break

    best_model = Trainer(sender, receiver)
    state = torch.load(
        "{}/best_model".format(run_folder),
        map_location=lambda storage, location: storage,
    )
    best_model.load_state_dict(state)
    best_model.to(device)
    # Evaluate best model on test data
    _, test_acc_meter, test_messages = evaluate(best_model, test_data)
    print("Test accuracy: {}".format(test_acc_meter.avg))

    # Update receiver and sender files with new state
    torch.save(best_model.sender, sender_file)
    torch.save(best_model.receiver, receiver_file)

    return test_acc_meter, test_messages
