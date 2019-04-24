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
from data.shapes import get_shapes_metadata, get_shapes_features
from data.obverter import get_obverter_metadata

from cee.metrics import representation_similarity_analysis, language_entropy

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
        help="type of input used by dataset pick from raw/features/meta (default meta)",
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

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
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
        )
        receiver = ObverterReceiver(
            args.vocab_size,
            embedding_size=args.embedding_size,
            dataset_type=args.dataset_type,
        )
    else:
        raise ValueError("Unsupported task type : {}".format(args.task))
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
            batch_size=args.batch_size,
            k=args.k,
            debug=args.debugging,
            dataset_type=args.dataset_type,
        )
        valid_meta_data = get_shapes_metadata(dataset="valid")
        valid_features = get_shapes_features(dataset="valid")
        eval_train_data = get_shapes_dataloader(
            batch_size=args.batch_size,
            k=args.k,
            debug=args.debugging,
            dataset="train",
            dataset_type=args.dataset_type,
        )

    elif args.task == "obverter":
        train_data, valid_data, test_data = get_obverter_dataloader(
            dataset_type=args.dataset_type,
            debug=args.debugging,
            batch_size=args.batch_size,
        )
        valid_meta_data = get_obverter_metadata(
            dataset="valid", first_picture_only=True
        )
        valid_features = None
        # eval train data is separate train dataloader to calculate
        # loss/acc on full set and get generalization error
        eval_train_data = get_obverter_dataloader(
            batch_size=args.batch_size,
            debug=args.debugging,
            dataset="train",
            dataset_type=args.dataset_type,
        )

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

    # early stopping with patience set to approx 10 epochs
    early_stopping = EarlyStopping(
        mode="max", patience=int((len(train_data) * 10) / args.log_interval)
    )

    # Train
    i = 0
    while i < args.iterations:
        for train_batch in train_data:
            loss, acc = train_one_batch(model, train_batch, optimizer)

            if i % args.log_interval == 0:
                loss_meter, acc_meter, _, _, _, _, = evaluate(model, eval_train_data)

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
                    writer.add_scalar("train_avg_loss", loss_meter.avg, i)
                    writer.add_scalar("train_avg_acc", acc_meter.avg, i)
                    writer.add_scalar("avg_loss", valid_loss_meter.avg, i)
                    writer.add_scalar("avg_acc", valid_acc_meter.avg, i)
                    writer.add_scalar("avg_entropy", valid_entropy_meter.avg, i)
                    writer.add_scalar("num_unique_messages", num_unique_messages, i)
                    writer.add_scalar("rsa_sr", rsa_sr, i)
                    writer.add_scalar("rsa_si", rsa_si, i)
                    writer.add_scalar("rsa_ri", rsa_ri, i)
                    writer.add_scalar("rsa_sm", rsa_sm, i)
                    writer.add_scalar(
                        "generalization_error", acc_meter.avg - valid_acc_meter.avg, i
                    )
                    writer.add_scalar(
                        "topological_similarity", topological_similarity, i
                    )
                    writer.add_scalar("pseudo_tre", pseudo_tre, i)
                    writer.add_scalar("language_entropy", l_entropy, i)

                early_stopping.step(valid_acc_meter.avg)
                if early_stopping.num_bad_epochs == 0:
                    torch.save(model.state_dict(), "{}/best_model".format(run_folder))

                # Skip for now
                print(
                    "{}/{} Iterations: average train loss: {}, average val loss: {}, \
                        average accuracy: {}, average val accuracy: {}".format(
                        i,
                        args.iterations,
                        loss_meter.avg,
                        valid_loss_meter.avg,
                        acc_meter.avg,
                        valid_acc_meter.avg,
                    )
                )

                if early_stopping.is_converged:
                    print("Converged in iterations {}".format(i))
                    break
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
