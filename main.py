# Cultural Evolution using Shapes

import argparse
import sys
import pickle
import torch
import warnings

from tensorboardX import SummaryWriter
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
        default=500000,
        metavar="N",
        help="number of batch iterations to train for (default: 500k)",
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
        "--cell-type",
        type=str,
        default="lstm",
        metavar="S",
        help="Cell in {'lstm', 'darts'} to select the cell type (default: lstm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 1024)",
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
        default=0.25,
        metavar="N",
        help="Percentage of population culled/mutated",
    )
    parser.add_argument(
        "--culling-mode",
        type=str,
        default="random",
        metavar="S",
        help="Mode in {'random', 'best', 'age'} to select which agent to reinitialize/cull (default: random)",
    )
    # Biological evolution
    parser.add_argument(
        "--evolution",
        help="Use evolution instead of random re-init (default: True)",
        action="store_true",
        default=False,
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
        args.embedding_size = 56

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
    writer = SummaryWriter(log_dir=experiment_folder + "/" + str(args.seed))

    # Load data
    train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k, debug=args.debugging
    )
    valid_meta_data = get_shapes_metadata(dataset="valid")
    valid_features = get_shapes_features(dataset="valid")

    # Generate population and save intial models
    shapes_cee = ShapesCEE(args, run_folder=experiment_folder)
    min_convergence_at_100, min_convergence_at_10 = 10, 10

    i = 0
    while i < args.iterations:
        for batch in train_data:
            shapes_cee.train_population(batch)
            if i % args.log_interval == 0:
                if i % args.metric_interval == 0:
                    advanced = True

                avg_loss, avg_acc, avg_entropy, rsa_sr, rsa_si, rsa_ri, topological_similarity, l_entropy, avg_unique = shapes_cee.evaluate_population(
                    valid_data, valid_meta_data, valid_features, advanced=advanced
                )
                writer.add_scalar("avg_acc", avg_acc, i)
                writer.add_scalar("avg_loss", avg_loss, i)
                writer.add_scalar("avg_entropy", avg_entropy, i)

                avg_age = shapes_cee.get_avg_age()
                writer.add_scalar("avg_age", avg_age, i)

                avg_convergence_at_10 = shapes_cee.get_avg_convergence_at_step(step=10)
                avg_convergence_at_100 = shapes_cee.get_avg_convergence_at_step(
                    step=100
                )

                writer.add_scalar("avg_convergence_at_10", avg_convergence_at_10, i)
                if avg_convergence_at_10 < min_convergence_at_10:
                    min_convergence_at_10 = avg_convergence_at_10
                writer.add_scalar("min_convergence_at_10", min_convergence_at_10, i)

                writer.add_scalar("avg_convergence_at_100", avg_convergence_at_100, i)
                if avg_convergence_at_100 < min_convergence_at_100:
                    min_convergence_at_100 = avg_convergence_at_100
                writer.add_scalar("min_convergence_at_100", min_convergence_at_100, i)

                writer.add_scalar("avg_unique_messages", avg_unique, i)

                if advanced == 0:
                    writer.add_scalar(
                        "topological_similarity", topological_similarity, i
                    )
                    writer.add_scalar("rsa_sr", rsa_sr, i)
                    writer.add_scalar("rsa_si", rsa_si, i)
                    writer.add_scalar("rsa_ri", rsa_ri, i)
                    writer.add_scalar("avg_language_entropy", l_entropy, i)
                    print(
                        "{0}/{1}\tAvg Loss: {2:.3g}\tAvg Acc: {3:.3g}\tAvg Age: {4:.3g}\tAvg Convergence: {5:.3g}\n\
                         Avg Entropy: {6:.3g} Avg RSA pS/R: {7:.3g}\tAvg RSA pS/I: {8:.3g}\tAvg RSA pR/I: {9:.3g}".format(
                            i,
                            args.iterations,
                            avg_loss,
                            avg_acc,
                            avg_age,
                            avg_convergence_at_100,
                            l_entropy,
                            rsa_sr,
                            rsa_si,
                            rsa_ri,
                        )
                    )
                else:
                    print(
                        "{0}/{1}\tAvg Loss: {2:.3g}\tAvg Acc: {3:.3g}\tAvg Age: {4:.3g}\tAvg Convergence: {5:.3g}".format(
                            i,
                            args.iterations,
                            avg_loss,
                            avg_acc,
                            avg_age,
                            avg_convergence_at_100,
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
                    shapes_cee.cull_population(
                        culling_rate=args.culling_rate, mode=args.culling_mode
                    )
                    # Cull receivers
                    shapes_cee.cull_population(
                        receiver=True,
                        culling_rate=args.culling_rate,
                        mode=args.culling_mode,
                    )
            i += 1


if __name__ == "__main__":
    main(sys.argv[1:])
