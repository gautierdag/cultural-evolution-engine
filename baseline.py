# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch

from tensorboardX import SummaryWriter

from utils import *

from cee.metrics import representation_similarity_analysis, language_entropy
from baseline_helper import get_sender_receiver, get_trainer, get_training_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender/Receiver Agent on a task"
    )
    parser.add_argument(
        "--debugging",
        help="Enable debugging mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--single-model",
        help="Use a single model (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="shapes",
        metavar="S",
        help="task to test on (default: shapes). Possible options: shapes or obverter",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="features",
        metavar="S",
        help="type of input used by dataset pick from raw/features/meta (default features)",
    )
    parser.add_argument(
        "--greedy",
        help="Use argmax at prediction time instead of sampling (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="number of batch iterations to train (default: 10k)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="number of iterations between logs (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        metavar="N",
        help="embedding size for embedding layer (default: 64)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
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
        default=5,
        metavar="N",
        help="max sentence length allowed for communication (default: 5)",
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
        default=5,
        metavar="N",
        help="Size of vocabulary (default: 5)",
    )
    parser.add_argument(
        "--darts",
        help="Use random architecture from DARTS space instead of random LSTMCell (default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=4,
        metavar="N",
        help="Size of darts cell to use with random-darts (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--sender-path",
        type=str,
        default=False,
        metavar="S",
        help="Sender to be loaded",
    )
    parser.add_argument(
        "--receiver-path",
        type=str,
        default=False,
        metavar="S",
        help="Receiver to be loaded",
    )
    parser.add_argument(
        "--freeze-sender",
        help="Freeze sender weights (do not train) ",
        action="store_true",
    )
    parser.add_argument(
        "--freeze-receiver",
        help="Freeze receiver weights (do not train) ",
        action="store_true",
    )
    parser.add_argument(
        "--obverter-setup",
        help="Enable obverter setup with shapes",
        action="store_true",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=False,
        metavar="S",
        help="Name to append to run file name",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=False,
        metavar="S",
        help="Additional folder within runs/",
    )
    parser.add_argument("--disable-print", help="Disable printing", action="store_true")

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5

    return args


def baseline(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    model_name = get_filename_from_baseline_params(args)
    if not args.folder:
        run_folder = "runs/" + model_name
    else:
        run_folder = "runs/" + args.folder + "/" + model_name

    writer = SummaryWriter(log_dir=run_folder + "/" + str(args.seed))
    train_data, valid_data, test_data, valid_meta_data, valid_features = get_training_data(
        args
    )

    # dump arguments
    pickle.dump(args, open("{}/experiment_params.p".format(run_folder), "wb"))

    # get sender and receiver models and save them
    sender, receiver = get_sender_receiver(args)

    sender_file = "{}/sender.p".format(run_folder)
    receiver_file = "{}/receiver.p".format(run_folder)
    torch.save(sender, sender_file)
    torch.save(receiver, receiver_file)

    model = get_trainer(sender, receiver, args)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    if not args.disable_print:
        # Print info
        print("----------------------------------------")
        print(
            "Model name: {} \n|V|: {}\nL: {}".format(
                model_name, args.vocab_size, args.max_length
            )
        )
        print(sender)
        print(receiver)
        print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_valid_acc = -1

    # Train
    i = 0
    running_loss = 0.0

    while i < args.iterations:
        for train_batch in train_data:

            loss, acc = train_one_batch(model, train_batch, optimizer)
            running_loss += loss

            if i % args.log_interval == 0:

                valid_loss_meter, valid_acc_meter, valid_entropy_meter, valid_messages, hidden_sender, hidden_receiver = evaluate(
                    model, valid_data
                )

                num_unique_messages = len(torch.unique(valid_messages, dim=0))
                valid_messages = valid_messages.cpu().numpy()

                rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity, pseudo_tre = representation_similarity_analysis(
                    valid_features,
                    valid_meta_data,
                    valid_messages,
                    hidden_sender,
                    hidden_receiver,
                    tre=True,
                )
                l_entropy = language_entropy(valid_messages)

                if writer is not None:
                    writer.add_scalar("avg_loss", valid_loss_meter.avg, i)
                    writer.add_scalar("avg_convergence", running_loss / (i + 1), i)
                    writer.add_scalar("avg_acc", valid_acc_meter.avg, i)
                    writer.add_scalar("avg_entropy", valid_entropy_meter.avg, i)
                    writer.add_scalar("avg_unique_messages", num_unique_messages, i)
                    writer.add_scalar("rsa_sr", rsa_sr, i)
                    writer.add_scalar("rsa_si", rsa_si, i)
                    writer.add_scalar("rsa_ri", rsa_ri, i)
                    writer.add_scalar("rsa_sm", rsa_sm, i)
                    writer.add_scalar(
                        "topological_similarity", topological_similarity, i
                    )
                    writer.add_scalar("pseudo_tre", pseudo_tre, i)
                    writer.add_scalar("language_entropy", l_entropy, i)

                if valid_acc_meter.avg > best_valid_acc:
                    best_valid_acc = valid_acc_meter.avg
                    torch.save(model.state_dict(), "{}/best_model".format(run_folder))

                metrics = {
                    "loss": valid_loss_meter.avg,
                    "acc": valid_acc_meter.avg,
                    "entropy": valid_entropy_meter.avg,
                    "l_entropy": l_entropy,
                    "rsa_sr": rsa_sr,
                    "rsa_si": rsa_si,
                    "rsa_ri": rsa_ri,
                    "rsa_sm": rsa_sm,
                    "pseudo_tre": pseudo_tre,
                    "topological_similarity": topological_similarity,
                    "num_unique_messages": num_unique_messages,
                    "avg_convergence": running_loss / (i + 1),
                }
                # dump metrics
                pickle.dump(
                    metrics, open("{}/metrics_at_{}.p".format(run_folder, i), "wb")
                )

                # Skip for now
                if not args.disable_print:
                    print(
                        "{}/{} Iterations: val loss: {}, val accuracy: {}".format(
                            i,
                            args.iterations,
                            valid_loss_meter.avg,
                            valid_acc_meter.avg,
                        )
                    )

            i += 1

    best_model = get_trainer(sender, receiver, args)
    state = torch.load(
        "{}/best_model".format(run_folder),
        map_location=lambda storage, location: storage,
    )
    best_model.load_state_dict(state)
    best_model.to(device)
    # Evaluate best model on test data
    _, test_acc_meter, _, test_messages, _, _ = evaluate(best_model, test_data)
    if not args.disable_print:
        print("Test accuracy: {}".format(test_acc_meter.avg))

    # Update receiver and sender files with new state
    torch.save(best_model.sender, sender_file)
    torch.save(best_model.receiver, receiver_file)

    if args.dataset_type == "raw":
        best_model.to(torch.device("cpu"))
        torch.save(best_model.visual_module, "data/extractor_{}.p".format(args.task))

    torch.save(test_messages, "{}/test_messages.p".format(run_folder))
    pickle.dump(
        test_acc_meter, open("{}/test_accuracy_meter.p".format(run_folder), "wb")
    )

    return run_folder


if __name__ == "__main__":
    baseline(sys.argv[1:])
