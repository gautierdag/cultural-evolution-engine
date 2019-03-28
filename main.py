# Cultural Evolution using Shapes

import argparse
import sys
import pickle
import torch
import warnings

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Receiver, Sender, Trainer
from utils import *
from data.shapes import ShapesVocab, get_shapes_dataset, get_shapes_metadata

from cee.population_utils import *
from cee.metrics import representation_similarity_analysis, language_entropy

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
        "--iterations",
        type=int,
        default=50000,
        metavar="N",
        help="number of batch iterations to train for (default: 50k)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        metavar="N",
        help="Number of iterations steps between evaluation (default: 1000)",
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
        "--culling-interval",
        type=int,
        default=5000,
        metavar="N",
        help="Number of sampling steps between culling (default: 5k)",
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
        filenames["senders"][sender_file] = {"avg_loss": 0, "avg_acc": 0, "age": 0}
        filenames["receivers"][receiver_file] = {"avg_loss": 0, "avg_acc": 0, "age": 0}

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


def evaluate_pair(sender_name, receiver_name, test_data):
    """
    Evaluates pair of sender/receiver on test data and returns avg loss/acc
    and generated messages
    Args:
        sender_name (path): path of sender model
        receiver_name (path): path of receiver model
        test_data (dataloader): dataloader of data to evaluate on
    Returns:
        avg_loss (float): average loss over data
        avg_acc (float): average accuracy over data
        test_messages (tensor): generated messages from data
    """
    sender = torch.load(sender_name)
    receiver = torch.load(receiver_name)
    model = Trainer(sender, receiver)
    model.to(device)
    test_loss_meter, test_acc_meter, test_messages = evaluate(model, test_data)

    return test_loss_meter.avg, test_acc_meter.avg, test_messages


def evaluate_messages(messages):
    metadata = get_shapes_metadata()
    messages = messages.cpu().numpy()
    rsa = representation_similarity_analysis(messages, metadata)
    l_entropy = language_entropy(messages)

    return rsa, l_entropy


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

    # Tensorboard tracker for evolution process
    timestamp = "/{:%m%d%H%M}".format(datetime.now())
    writer = SummaryWriter(log_dir=experiment_folder + "/" + timestamp)

    # Load data
    train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k, debug=args.debugging
    )

    i = 0
    while i < args.iterations:
        for batch in train_data:
            # Sampling from population
            sender_name = sample_population(population_filenames["senders"])
            receiver_name = sample_population(population_filenames["receivers"])
            loss, acc = shapes_trainer(sender_name, receiver_name, batch)

            if i % args.log_interval == 0:
                if args.debugging:
                    avg_loss, avg_acc, test_messages = evaluate_pair(
                        sender_name, receiver_name, valid_data
                    )
                else:
                    avg_loss, avg_acc, test_messages = evaluate_pair(
                        sender_name, receiver_name, test_data
                    )

                print(
                    "{0}/{1}\tTest Loss: {2:.3g}\tTest Acc: {3:.3g}".format(
                        i, args.iterations, avg_loss, avg_acc
                    )
                )
                rsa, l_entropy = evaluate_messages(test_messages)

                writer.add_scalar("rsa", rsa, i)
                writer.add_scalar("language_entropy", l_entropy, i)
                writer.add_scalar("avg_acc", avg_acc, i)
                writer.add_scalar("avg_loss", avg_loss, i)

                torch.save(
                    test_messages, "{}/test_messages_{}.p".format(experiment_folder, i)
                )

            if i % args.culling_interval == 0 and i > 0:
                cull_population(population_filenames["senders"])
                cull_population(population_filenames["receivers"])

            i += 1


if __name__ == "__main__":
    cee(sys.argv[1:])
