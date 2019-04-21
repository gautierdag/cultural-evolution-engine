# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch

from tensorboardX import SummaryWriter
from model import (
    ShapesReceiver,
    ShapesSender,
    ShapesTrainer,
    ObverterReceiver,
    ObverterSender,
    ObverterTrainer,
    generate_genotype,
)
from utils import *
from data import AgentVocab, get_shapes_dataloader, get_obverter_dataloader

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
        "--greedy",
        help="Use argmax at prediction time instead of sampling (default: False)",
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
        "--embedding-size",
        type=int,
        default=256,
        metavar="N",
        help="embedding size for embedding layer (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
    parser.add_argument(
        "--darts",
        help="Use random architecture from DARTS space instead of random LSTMCell (default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=8,
        metavar="N",
        help="Size of darts cell to use with random-darts (default: 8)",
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
        default="meta",
        metavar="S",
        help="type of input used by dataset pick from raw/features/meta/combined (default meta)",
    )

    args = parser.parse_args(args)

    args.color_vocab_size = None
    args.object_vocab_size = None

    if args.debugging:
        args.epochs = 10
        args.max_length = 5

    return args


def get_sender_receiver(args):
    # Load Vocab
    vocab = AgentVocab(args.vocab_size)

    if args.task == "shapes":
        cell_type = "lstm"
        genotype = {}
        if args.darts:
            cell_type = "darts"
            genotype = generate_genotype(num_nodes=args.num_nodes)
            print(genotype)

        sender = ShapesSender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            greedy=args.greedy,
            cell_type=cell_type,
            genotype=genotype,
        )
        receiver = ShapesReceiver(
            args.vocab_size,
            embedding_size=args.embedding_size,
            cell_type=cell_type,
            genotype=genotype,
        )
    elif args.task == "obverter":
        sender = ObverterSender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            greedy=args.greedy,
            dataset_type=args.dataset_type,
            object_vocab_size=args.object_vocab_size,
            color_vocab_size=args.color_vocab_size,
        )
        receiver = ObverterReceiver(
            args.vocab_size,
            embedding_size=args.embedding_size,
            dataset_type=args.dataset_type,
            object_vocab_size=args.object_vocab_size,
            color_vocab_size=args.color_vocab_size,
        )
    else:
        raise ValueError("Unsupported task type : {}".formate(args.task))
    return sender, receiver


def get_trainer(sender, receiver, args):
    if args.task == "shapes":
        return ShapesTrainer(sender, receiver)
    if args.task == "obverter":
        return ObverterTrainer(sender, receiver)


def baseline(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    model_name = get_filename_from_baseline_params(args)
    run_folder = "runs/" + model_name
    writer = SummaryWriter(log_dir=run_folder + "/" + str(args.seed))

    # Load data
    if args.task == "shapes":
        train_data, valid_data, test_data = get_shapes_dataloader(
            batch_size=args.batch_size, k=args.k, debug=args.debugging
        )
    elif args.task == "obverter":
        train_data, valid_data, test_data, meta_vocabs = get_obverter_dataloader(
            dataset_type=args.dataset_type,
            debug=args.debugging,
            batch_size=args.batch_size,
        )
        if args.dataset_type == "meta" or args.dataset_type == "combined":
            args.color_vocab_size = len(meta_vocabs[0].itos)
            args.object_vocab_size = len(meta_vocabs[1].itos)
    else:
        raise ValueError("Unsupported task type : {}".formate(args.task))

    # Print info
    print("----------------------------------------")
    print(
        "Model name: {} \n|V|: {}\nL: {}".format(
            model_name, args.vocab_size, args.max_length
        )
    )
    # get sender and receiver models and save them
    sender, receiver = get_sender_receiver(args)
    print(sender)
    print(receiver)
    sender_file = "{}/sender.p".format(run_folder)
    receiver_file = "{}/receiver.p".format(run_folder)
    torch.save(sender, sender_file)
    torch.save(receiver, receiver_file)

    model = get_trainer(sender, receiver, args)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(mode="max")

    # Train
    for epoch in range(args.epochs):
        loss_meter, acc_meter = train_one_epoch(model, train_data, optimizer)
        eval_loss_meter, eval_acc_meter, entropy_meter, eval_messages, _, _ = evaluate(
            model, valid_data
        )

        if writer is not None:
            writer.add_scalar("avg_train_loss", loss_meter.avg, epoch)
            writer.add_scalar("avg_train_acc", acc_meter.avg, epoch)
            writer.add_scalar("avg_loss", eval_loss_meter.avg, epoch)
            writer.add_scalar("avg_acc", eval_acc_meter.avg, epoch)
            writer.add_scalar("avg_entropy", entropy_meter.avg, epoch)

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

    best_model = get_trainer(sender, receiver, args)
    state = torch.load(
        "{}/best_model".format(run_folder),
        map_location=lambda storage, location: storage,
    )
    best_model.load_state_dict(state)
    best_model.to(device)
    # Evaluate best model on test data
    _, test_acc_meter, _, test_messages, _, _ = evaluate(best_model, test_data)
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
