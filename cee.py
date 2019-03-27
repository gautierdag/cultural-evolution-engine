# Baseline setting in which there are only two agents
# - no evolution

import argparse
import sys
import pickle
import torch
import warnings

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Receiver, Sender, Trainer
from utils import *
from data.shapes import ShapesVocab, get_shapes_dataset
from cee.population_utils import *

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
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="Number of iterations steps between evaluation",
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
        default=3,
        metavar="N",
        help="Size of each sender and receiver pop (default: 4)",
    )
    parser.add_argument(
        "--culling-interval",
        type=int,
        default=4,
        metavar="N",
        help="Number of sampling steps between culling",
    )
    parser.add_argument(
        "--culling-rate",
        type=float,
        default=0.2,
        metavar="N",
        help="Percentage of population culled",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.epochs = 10
        args.max_length = 5

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
    filenames = {"senders": {}, "receivers": {}}
    create_folder_if_not_exists(run_folder + "/senders")
    create_folder_if_not_exists(run_folder + "/receivers")

    # Load Vocab
    vocab = ShapesVocab(args.vocab_size)

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
        filenames["senders"][sender_file] = 0
        filenames["receivers"][receiver_file] = 0
    return filenames


def shapes_trainer(sender_name, receiver_name, batch):
    """
    Trains sender and receiver model for one batch
    Args:
        sender_name (path): path of sender model
        receiver_name (path): path of receiver model
        batch: batch from dataloader
    Returns:
        loss: loss from batch
        acc: accuracy from batch
    """
    sender = torch.load(sender_name)
    receiver = torch.load(receiver_name)

    model = Trainer(sender, receiver)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss, acc = train_one_batch(model, batch, optimizer)

    # Update receiver and sender files with new state
    torch.save(model.sender, sender_name)
    torch.save(model.receiver, receiver_name)

    return loss, acc


def cee(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Generate name for experiment folder
    experiment_name = get_filename_from_cee_params(args)
    experiment_folder = "runs/" + experiment_name

    # Create Experiment folder if doesn't exist
    create_folder_if_not_exists(experiment_folder)

    # Save experiment params
    pickle.dump(args, open("{}/experiment_params.p".format(experiment_folder), "wb"))

    # Generate population and save intial models
    population_filenames = initialize_models(args, run_folder=experiment_folder)

    i = 0
    for epoch in range(args.epochs):

        # Load data
        train_data, valid_data, test_data = get_shapes_dataset(
            batch_size=args.batch_size, k=args.k, debug=args.debugging
        )

        for batch in train_data:
            # Sampling from population
            sender_name = sample_population(population_filenames["senders"])
            receiver_name = sample_population(population_filenames["receivers"])
            loss, acc = shapes_trainer(sender_name, receiver_name, batch)
            print("Loss: {0:.3g} \t Acc: {1:.3g}".format(loss, acc))
            if i % args.log_interval == 0 and i > 0:
                print("Evaluating Match")

            i += 1

        cull_population(population_filenames["senders"])
        cull_population(population_filenames["receivers"])


if __name__ == "__main__":
    cee(sys.argv[1:])
