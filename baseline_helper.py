import torch
import numpy as np

from data import AgentVocab, get_shapes_dataloader, get_obverter_dataloader
from data.shapes import get_shapes_metadata, get_shapes_features
from data.obverter import get_obverter_metadata, get_obverter_features


from model import (
    ShapesReceiver,
    ShapesSender,
    ShapesTrainer,
    ObverterReceiver,
    ObverterSender,
    ObverterTrainer,
    generate_genotype,
    ShapesSingleModel,
    ShapesMetaVisualModule,
    ObverterSingleModel,
    ObverterMetaVisualModule,
)


def get_sender_receiver(args):
    # Load Vocab
    vocab = AgentVocab(args.vocab_size)
    cell_type = "lstm"
    genotype = {}
    if args.darts:
        cell_type = "darts"
        genotype = generate_genotype(num_nodes=args.num_nodes)
        if not args.disable_print:
            print(genotype)

    if args.task == "shapes" and not args.obverter_setup:
        if args.single_model:
            sender = ShapesSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ShapesSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
        else:
            sender = ShapesSender(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ShapesReceiver(
                args.vocab_size,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
    elif args.task == "obverter" or (args.obverter_setup and args.task == "shapes"):
        if args.single_model:
            sender = ObverterSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ObverterSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
        else:
            sender = ObverterSender(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ObverterReceiver(
                args.vocab_size,
                hidden_size=args.hidden_size,
                embedding_size=args.embedding_size,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
    else:
        raise ValueError("Unsupported task type : {}".format(args.task))

    if args.sender_path:
        sender = torch.load(args.sender_path)
    if args.receiver_path:
        receiver = torch.load(args.receiver_path)

    if args.task == "shapes":
        meta_vocab_size = 15
    else:
        meta_vocab_size = 13

    if args.task == "obverter" or (args.obverter_setup and args.task == "shapes"):
        if args.freeze_sender:
            for param in sender.parameters():
                param.requires_grad = False
        else:
            s_visual_module = ObverterMetaVisualModule(
                hidden_size=sender.hidden_size,
                dataset_type=args.dataset_type,
                meta_vocab_size=meta_vocab_size,
            )
            sender.input_module = s_visual_module
            sender.reset_parameters()
        if args.freeze_receiver:
            for param in receiver.parameters():
                param.requires_grad = False
        else:
            r_visual_module = ObverterMetaVisualModule(
                hidden_size=receiver.hidden_size,
                dataset_type=args.dataset_type,
                meta_vocab_size=meta_vocab_size,
            )
            receiver.input_module = r_visual_module
            receiver.reset_parameters()

    if args.task == "shapes" and not args.obverter_setup:
        if args.freeze_sender:
            for param in sender.parameters():
                param.requires_grad = False
        else:
            s_visual_module = ShapesMetaVisualModule(
                hidden_size=sender.hidden_size, dataset_type=args.dataset_type
            )
            sender.input_module = s_visual_module
        if args.freeze_receiver:
            for param in receiver.parameters():
                param.requires_grad = False
        else:
            r_visual_module = ShapesMetaVisualModule(
                hidden_size=receiver.hidden_size,
                dataset_type=args.dataset_type,
                sender=False,
            )

            if args.single_model:
                receiver.output_module = r_visual_module
            else:
                receiver.input_module = r_visual_module

    return sender, receiver


def get_trainer(sender, receiver, args):
    extract_features = args.dataset_type == "raw"
    if args.task == "shapes" and not args.obverter_setup:
        return ShapesTrainer(sender, receiver, extract_features=extract_features)
    if args.task == "obverter" or (args.obverter_setup and args.task == "shapes"):
        return ObverterTrainer(sender, receiver, extract_features=extract_features)


def get_training_data(args):
    # Load data
    if args.task == "shapes":
        train_data, valid_data, test_data = get_shapes_dataloader(
            batch_size=args.batch_size,
            k=args.k,
            debug=args.debugging,
            dataset_type=args.dataset_type,
            obverter_setup=args.obverter_setup,
        )
        valid_meta_data = get_shapes_metadata(dataset="valid")
        valid_features = get_shapes_features(dataset="valid")

    elif args.task == "obverter":
        train_data, valid_data, test_data = get_obverter_dataloader(
            dataset_type=args.dataset_type,
            debug=args.debugging,
            batch_size=args.batch_size,
        )
        valid_meta_data = get_obverter_metadata(
            dataset="valid", first_picture_only=True
        )
        valid_features = get_obverter_features(dataset="valid")

    else:
        raise ValueError("Unsupported task type : {}".formate(args.task))

    return (train_data, valid_data, test_data, valid_meta_data, valid_features)


def get_raw_data(args, dataset="valid"):
    if args.task == "shapes":
        valid_raw = get_shapes_features(dataset=dataset, mode="raw")
        return valid_raw
    else:
        raise ValueError("Unsupported task type for raw : {}".formate(args.task))


def save_example_images(args, filename):
    if args.save_example_batch:
        valid_raw = get_raw_data(args)
        valid_raw = valid_raw
        file_path = filename + "/example_batch.npy"
        np.save(file_path, valid_raw)
        valid_meta = get_shapes_metadata(dataset="valid")

        file_path = filename + "/example_batch_meta.npy"
        np.save(file_path, valid_meta)
