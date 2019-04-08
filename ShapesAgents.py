from cee import BaseAgent
from model import Sender, Receiver
from data.shapes import ShapesVocab

import torch


class SenderAgent(BaseAgent):
    def __init__(self, filename, args, genotype=None):
        if args.cell_type == "darts" and genotype is None:
            raise ValueError("Expected genotype in Sender with option 'darts'")

        super().__init__(filename, args)
        self.genotype = genotype

        vocab = ShapesVocab(args.vocab_size)
        sender = Sender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
            cell_type=args.cell_type,
            genotype=genotype,
        )
        torch.save(sender, filename)


class ReceiverAgent(BaseAgent):
    def __init__(self, filename, args):
        super().__init__(filename, args)
        receiver = Receiver(
            args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
        )
        torch.save(receiver, filename)
