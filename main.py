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
        "--debugging",
        help="Enable debugging mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        help="Resumes from checkpoint (if present) (default: False)",
        action="store_true",
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
        default=5000,
        metavar="N",
        help="Number of iterations steps between evaluation (default: 500)",
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
        default=16,
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
        args.log_interval = 500
        args.population_size = 4

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
    eval_train_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k, debug=args.debugging, dataset="train"
    )
    valid_meta_data = get_shapes_metadata(dataset="valid")
    valid_features = get_shapes_features(dataset="valid")

    # Generate population and save intial models
    if args.resume:
        print("Resuming from checkpoint")
        shapes_cee = pickle.load(open(experiment_folder + "/cee.p", "rb"))
        i = shapes_cee.iteration
    else:
        shapes_cee = ShapesCEE(args, run_folder=experiment_folder)
        i = 0

    while i < args.iterations:
        for batch in train_data:
            shapes_cee.train_population(batch)
            shapes_cee.save()
            if i % args.log_interval == 0:
                metrics = shapes_cee.evaluate_population(
                    valid_data, valid_meta_data, valid_features, advanced=True
                )
                # evaluate on full train set (using separate dataloader)
                train_metrics = shapes_cee.evaluate_population(
                    eval_train_data, [], [], advanced=False
                )

                writer.add_scalar("avg_acc", metrics["acc"], i)
                writer.add_scalar("avg_loss", metrics["loss"], i)
                writer.add_scalar("avg_entropy", metrics["entropy"], i)
                writer.add_scalar("train_avg_acc", train_metrics["acc"], i)
                writer.add_scalar("train_avg_loss", train_metrics["loss"], i)
                writer.add_scalar("train_avg_entropy", train_metrics["entropy"], i)
                writer.add_scalar(
                    "generalization_error", train_metrics["acc"] - metrics["acc"], i
                )

                metrics["avg_age"] = shapes_cee.get_avg_age()
                writer.add_scalar("avg_age", metrics["avg_age"], i)

                # convergence calculations
                metrics["avg_convergence"] = shapes_cee.get_avg_convergence_at_step(
                    dynamic=True
                )
                writer.add_scalar("avg_convergence", metrics["avg_convergence"], i)

                metrics[
                    "avg_convergence_at_10"
                ] = shapes_cee.get_avg_convergence_at_step(step=10)

                metrics[
                    "avg_convergence_at_100"
                ] = shapes_cee.get_avg_convergence_at_step(step=100)

                writer.add_scalar(
                    "avg_convergence_at_10", metrics["avg_convergence_at_10"], i
                )
                writer.add_scalar(
                    "avg_convergence_at_100", metrics["avg_convergence_at_100"], i
                )

                writer.add_scalar(
                    "avg_unique_messages", metrics["num_unique_messages"], i
                )

                writer.add_scalar(
                    "topological_similarity", metrics["topological_similarity"], i
                )
                writer.add_scalar("rsa_sr", metrics["rsa_sr"], i)
                writer.add_scalar("rsa_si", metrics["rsa_si"], i)
                writer.add_scalar("rsa_ri", metrics["rsa_ri"], i)

                writer.add_scalar("avg_language_entropy", metrics["l_entropy"], i)
                writer.add_scalar("avg_message_dist", metrics["avg_message_dist"], i)
                writer.add_scalar("avg_matches", metrics["avg_matches"], i)

                print(
                    "{0}/{1}\tAvg Loss: {2:.3g}\tAvg Acc: {3:.3g}\tAvg Age: {4:.3g}\tAvg Convergence: {5:.3g}\n\
                        Avg Entropy: {6:.3g} Avg RSA pS/R: {7:.3g}\tAvg RSA pS/I: {8:.3g}\tAvg RSA pR/I: {9:.3g}".format(
                        i,
                        args.iterations,
                        metrics["loss"],
                        metrics["acc"],
                        metrics["avg_age"],
                        metrics["avg_convergence_at_100"],
                        metrics["l_entropy"],
                        metrics["rsa_sr"],
                        metrics["rsa_si"],
                        metrics["rsa_ri"],
                    )
                )

                # dump metrics
                pickle.dump(
                    metrics,
                    open("{}/metrics_at_{}.p".format(experiment_folder, i), "wb"),
                )
                pickle.dump(
                    train_metrics,
                    open("{}/train_metrics_at_{}.p".format(experiment_folder, i), "wb"),
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
