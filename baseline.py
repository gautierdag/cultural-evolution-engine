# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch

from tensorboardX import SummaryWriter
from datetime import datetime
from model import Receiver, Sender, Trainer
from utils import *
from data.shapes import ShapesVocab, get_shapes_dataset

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

    args = parser.parse_args(args)

    if args.debugging:
        args.epochs = 10
        args.max_length = 5

    return args


def baseline(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    # Load Vocab
    vocab = ShapesVocab(args.vocab_size)

    model_name = get_filename_from_baseline_params(args)
    timestamp = "/{:%m%d%H%M}".format(datetime.now())
    run_folder = "runs/" + model_name
    writer = SummaryWriter(log_dir=run_folder + "/" + timestamp)

    # Print info
    print("----------------------------------------")
    print(
        "Model name: {} \n|V|: {}\nL: {}".format(
            model_name, args.vocab_size, args.max_length
        )
    )
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
    print(sender)
    print(receiver)

    sender_file = "{}/sender.p".format(run_folder)
    receiver_file = "{}/receiver.p".format(run_folder)
    torch.save(sender, sender_file)
    torch.save(receiver, receiver_file)

    # Load data
    train_data, valid_data, test_data = get_shapes_dataset(
        batch_size=args.batch_size, k=args.k, debug=args.debugging
    )

    sender = torch.load(sender_file)
    receiver = torch.load(receiver_file)
    model = Trainer(sender, receiver)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(mode="max")

    # Train
    for epoch in range(args.epochs):
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

    torch.save(test_messages, "{}/test_messages.p".format(run_folder))
    pickle.dump(
        test_acc_meter, open("{}/test_accuracy_meter.p".format(run_folder), "wb")
    )


if __name__ == "__main__":
    baseline(sys.argv[1:])
