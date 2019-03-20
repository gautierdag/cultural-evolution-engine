# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import os
import sys
import torch
import warnings

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Receiver, Sender
from train_utils import *
from data.shapes import ShapesVocab
from shapes_trainer import shapes_trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender Receiver Agent on Shapes"
    )
    parser.add_argument(
        "--debugging", help="Enable debugging mode (default: False", action="store_true"
    )
    parser.add_argument(
        "--greedy",
        help="Use argmax at prediction time instead of sampling (default: False",
        action="store_true",
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
    # Cultural evolution parameters
    parser.add_argument(
        "--population-size",
        type=int,
        default=4,
        metavar="N",
        help="Size of each sender and receiver pop (default: 4)",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=20,
        metavar="N",
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--culling-interval",
        type=int,
        default=4,
        metavar="N",
        help="Number of sampling steps between culling",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.epochs = 10
        args.max_length = 5

    if args.sampling_steps <= args.culling_interval:
        warnings.warn(
            "Culling interval greater than sampling steps.\n \
            This means population will never be culled!"
        )

    return args


def initialize_models(args, run_folder="runs/"):
    """
    Initializes args.population_size sender and receiver models
    Args:
        args (required): arguments obtained from argparse
        run_folder (req): path of run folder to save models in
    Returns:
        filenames (dict): dictionary containing the filepaths of the senders and receivers
    """
    filenames = {"senders": [], "receivers": []}
    for i in range(args.population_size):
        sender = Sender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
        )
        receiver = Receiver(
            args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
        )
        sender_file = "{}/senders/sender_{}.p".format(run_folder, i)
        receiver_file = "{}/receivers/receiver_{}.p".format(run_folder, i)
        torch.save(sender, sender_file)
        torch.save(receiver, receiver_file)
        filenames["senders"].append(sender_file)
        filenames["senders"].append(receiver_file)
    return filenames


def cee(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Load Vocab
    vocab = ShapesVocab(args.vocab_size)

    # Generate name for experiment folder
    experiment_folder = get_filename_from_cee_params(args)

    timestamp = "/{:%m%d%H%M}".format(datetime.now())
    run_folder = "runs/" + model_name
    writer = SummaryWriter(log_dir=run_folder + "/" + timestamp)
    population_filenames = initialize_models(args)


if __name__ == "__main__":
    cee(sys.argv[1:])
