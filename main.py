# Cultural Evolution using Shapes

import argparse
import sys
import pickle
import torch
import warnings

from tensorboardX import SummaryWriter
from datetime import datetime
from utils import *
from data.shapes import get_shapes_dataset, get_shapes_metadata, get_shapes_features

from ShapesCEE import ShapesCEE

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
        "--run_folder", type=str, default="runs", help="Run folder path (default: runs)"
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
        default=500,
        metavar="N",
        help="Number of iterations steps between evaluation (default: 500)",
    )
    parser.add_argument(
        "--metric-interval",
        type=int,
        default=2500,
        metavar="N",
        help="Number of iterations steps between more advanced metrics calculations (default: 2500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
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
    # Simple Cultural evolution parameters
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
        help="Number of sampling steps between culling/mutate (default: 5k)",
    )
    parser.add_argument(
        "--culling-rate",
        type=float,
        default=0.2,
        metavar="N",
        help="Percentage of population culled/mutated",
    )
    # Biological evolution
    parser.add_argument(
        "--evolution",
        help="Use evolution instead of random re-init (default: True)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--init-nodes",
        type=int,
        default=1,
        metavar="N",
        help="Initial number of nodes in DARTs cell (default: 1)",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 10000
        args.culling_interval = 2000
        args.max_length = 5

    if args.evolution:
        args.cell_type = "darts"

    return args


def main(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Generate name for experiment folder
    experiment_name = get_filename_from_cee_params(args)
    experiment_folder = "runs/" + experiment_name

    # Create Experiment folder if doesn't exist
    create_folder_if_not_exists(experiment_folder)

    # Save experiment params
    pickle.dump(args, open("{}/experiment_params.p".format(experiment_folder), "wb"))

    # Tensorboard tracker for evolution process
    timestamp = "/{:%m%d%H%M}".format(datetime.now())
    writer = SummaryWriter(log_dir=experiment_folder + "/" + timestamp)

    # Load data
    train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k, debug=args.debugging
    )
    valid_meta_data = get_shapes_metadata(dataset="valid")
    valid_features = get_shapes_features(dataset="valid")

    # Generate population and save intial models
    shapes_cee = ShapesCEE(args, run_folder=experiment_folder)

    i = 0
    while i < args.iterations:
        for batch in train_data:
            shapes_cee.train_population(batch)
            if i % args.log_interval == 0:
                avg_loss, avg_acc, rsa_sr, rsa_si, rsa_ri, topological_similarity, l_entropy = shapes_cee.evaluate_population(
                    valid_data, valid_meta_data, valid_features
                )
                avg_age = shapes_cee.get_avg_age()
                avg_speed = shapes_cee.get_avg_speed()
                writer.add_scalar("avg_acc", avg_acc, i)
                writer.add_scalar("avg_loss", avg_loss, i)
                writer.add_scalar("avg_age", avg_age, i)
                writer.add_scalar("avg_speed", avg_speed, i)
                if i % args.metric_interval == 0:
                    writer.add_scalar(
                        "topological_similarity", topological_similarity, i
                    )
                    writer.add_scalars(
                        "rsa", {"pS/R": rsa_sr, "pS/I": rsa_si, "pR/I": rsa_ri}, i
                    )
                    writer.add_scalar("language_entropy", l_entropy, i)
                    print(
                        "{0}/{1}\tAvg Loss: {2:.3g}\tAvg Acc: {3:.3g}\tAvg Entropy: {4:.3g}\tAvg RSA \
                        pS/R: {5:.3g}\tAvg RSA pS/I: {6:.3g}\tAvg RSA pR/I: {7:.3g}".format(
                            i,
                            args.iterations,
                            avg_loss,
                            avg_acc,
                            l_entropy,
                            rsa_sr,
                            rsa_si,
                            rsa_ri,
                        )
                    )
                else:
                    print(
                        "{0}/{1}\tAvg Loss: {2:.3g}\tAvg Acc: {3:.3g}\tAvg Age: {4:.3g}\tAvg Speed: {5:.3g}".format(
                            i, args.iterations, avg_loss, avg_acc, avg_age, avg_speed
                        )
                    )

            if i % args.culling_interval == 0 and i > 0:
                if args.evolution:
                    shapes_cee.mutate_population(culling_rate=args.culling_rate)
                    shapes_cee.mutate_population(
                        culling_rate=args.culling_rate, receiver=True
                    )
                else:
                    # Cull senders
                    shapes_cee.cull_population(culling_rate=args.culling_rate)
                    # Cull receivers
                    shapes_cee.cull_population(
                        receiver=True, culling_rate=args.culling_rate
                    )
            i += 1


if __name__ == "__main__":
    main(sys.argv[1:])
