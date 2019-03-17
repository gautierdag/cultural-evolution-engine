import pickle
import numpy as np
import random
import argparse
import os
import sys
import torch

from models import Model
from train_utils import train_one_epoch, evaluate, EarlyStopping
from data.shapes import get_shapes_dataset, ShapesVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=42):
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender Receiver Agent on Shapes"
    )
    parser.add_argument(
        "--debugging", help="Enable debugging mode (default: False", action="store_true"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        metavar="N",
        help="hidden size for RNN encoder (default: 512)",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=256,
        metavar="N",
        help="embedding size for embedding layer (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        metavar="N",
        help="max sentence length allowed for communication (default: 10)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        metavar="N",
        help="Number of distractors (default: 3)",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10,
        metavar="N",
        help="Size of vocabulary (default: 10)",
    )

    args = parser.parse_args(args)

    return args


def main(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Load Vocab
    vocab = ShapesVocab(args.vocab_size)

    # Load data
    n_image_features, train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k
    )

    model_id = "baseline"

    # Print info
    print("----------------------------------------")
    print("Model name: {}".format(model_id))
    print("|V|: {}".format(args.vocab_size))
    print("L: {}".format(args.max_length))

    model = Model(
        n_image_features,
        args.vocab_size,
        args.embedding_size,
        args.hidden_size,
        args.batch_size,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(mode="max")

    losses_meters = []
    eval_losses_meters = []
    accuracy_meters = []
    eval_accuracy_meters = []

    # Train
    for epoch in range(args.epochs):

        epoch_loss_meter, epoch_acc_meter = train_one_epoch(
            model, train_data, optimizer, vocab.bound_idx, args.max_length
        )

        losses_meters.append(epoch_loss_meter)
        accuracy_meters.append(epoch_acc_meter)

        eval_loss_meter, eval_acc_meter, eval_messages = evaluate(
            model, valid_data, vocab.bound_idx, args.max_length
        )

        early_stopping.step(eval_acc_meter.avg)

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

        if early_stopping.is_converged:
            print("Converged in epoch {}".format(epoch))
            break

    # Evaluate best model on test data
    _, test_acc_meter, test_messages = evaluate(
        model, test_data, vocab.bound_idx, args.max_length
    )
    print("Test accuracy: {}".format(test_acc_meter.avg))


if __name__ == "__main__":
    main(sys.argv[1:])
