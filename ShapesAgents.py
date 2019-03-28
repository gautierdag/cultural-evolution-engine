from cee import BaseAgent
from model import Sender, Receiver
from data.shapes import ShapesVocab

import torch


class SenderAgent(BaseAgent):
    def __init__(self, filename, args):
        super().__init__(filename, args)
        vocab = ShapesVocab(args.vocab_size)
        sender = Sender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
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
