# Baseline setting in which there are only two agents
# - no evolution

import argparse
import sys
import torch
import warnings

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Receiver, Sender
from utils import *
from data.shapes import ShapesVocab

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
        default=3,
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
        filenames["senders"].append(sender_file)
        filenames["receivers"].append(receiver_file)
    return filenames


def cull_model(model_filepath):
    """
    Reinitialize the weights of a single model
    """
    model = torch.load(model_filepath)
    model.reset_parameters()
    torch.save(model, model_filepath)


def cee(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Load Vocab
    vocab = ShapesVocab(args.vocab_size)

    # Generate name for experiment folder
    experiment_name = get_filename_from_cee_params(args)
    experiment_folder = "runs/" + experiment_name

    # Create Experiment folder if doesn't exist
    create_folder_if_not_exists(experiment_folder)
    population_filenames = initialize_models(args, run_folder=experiment_folder)

    for i in range(args.sampling_steps):
        # Sampling from population
        s_r = random.randrange(0, args.population_size)
        r_r = random.randrange(0, args.population_size)

        print("Matching sender {} with  receiver {}".format(s_r, r_r))
        sender_name = population_filenames["senders"][s_r]
        receiver_name = population_filenames["receivers"][r_r]

        match_folder_path = experiment_folder + "/s{}_r{}".format(s_r, r_r)
        create_folder_if_not_exists(match_folder_path)

        timestamp = "/{:%m%d%H%M}".format(datetime.now())
        writer = SummaryWriter(log_dir=match_folder_path + "/" + timestamp)
        test_acc_meter, test_messages = shapes_trainer(
            args,
            sender_name,
            receiver_name,
            writer=writer,
            run_folder=experiment_folder,  # used for best_model
        )
        torch.save(
            test_messages, "{}/test_messages_step_{}.p".format(experiment_folder, i)
        )
        pickle.dump(
            test_acc_meter,
            open("{}/test_accuracy_meter_{}.p".format(match_folder_path, i), "wb"),
        )

        if i % args.culling_interval == 0 and i != 0:
            c = max(1, int(args.culling_rate * args.population_size))
            print("Culling {} models from sender and receiver populations".format(c))
            for _ in range(c):
                s_r = random.randrange(0, args.population_size)
                r_r = random.randrange(0, args.population_size)
                sender_name = population_filenames["senders"][s_r]
                receiver_name = population_filenames["receivers"][r_r]
                cull_model(sender_name)
                cull_model(sender_name)


if __name__ == "__main__":
    cee(sys.argv[1:])
