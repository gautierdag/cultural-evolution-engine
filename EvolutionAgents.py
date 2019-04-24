from cee import BaseAgent
from model import (
    ShapesSender,
    ShapesReceiver,
    ObverterSender,
    ObverterReceiver,
    get_genotype_image,
)
from data import AgentVocab

import torch
import pickle


class SenderAgent(BaseAgent):
    def __init__(self, run_folder, args, genotype=None, agent_id=0):
        if args.cell_type == "darts" and genotype is None:
            raise ValueError("Expected genotype in Sender with option 'darts'")

        self.run_folder = run_folder
        self.agent_id = agent_id
        filename = "{}/senders/sender_{}.p".format(run_folder, agent_id)

        super().__init__(filename, args)
        self.genotype = genotype
        self.convergence = 100

        vocab = AgentVocab(args.vocab_size)
        self.vocab_bound_idx = vocab.bound_idx

        sender = self.get_sender(genotype)
        torch.save(sender, filename)

    def get_sender(self, genotype):
        if args.task == "shapes":
            sender = ShapesSender(
                args.vocab_size,
                args.max_length,
                self.vocab_bound_idx,
                embedding_size=args.embedding_size,
                greedy=args.greedy,
                cell_type=args.cell_type,
                genotype=genotype,
            )
        if args.task == "obverter":
            sender = ObverterSender(
                args.vocab_size,
                args.max_length,
                self.vocab_bound_idx,
                embedding_size=args.embedding_size,
                greedy=args.greedy,
                cell_type=args.cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
                meta_vocab_size=args.meta_vocab_size,
            )
        return sender

    def mutate(self, new_genotype):
        """
        Mutate model to new genotype
        """
        self.genotype = new_genotype

        vocab = AgentVocab(self.args.vocab_size)

        model = get_sender()
        torch.save(model, self.filename)

        self.initialize_loss_acc()
        self.age = 0

    def save_genotype(self, generation=0, metrics={}):
        geno_filename = "{}/senders_genotype/sender_{}_generation_{}".format(
            self.run_folder, self.agent_id, generation
        )
        img = get_genotype_image(
            self.genotype.recurrent, geno_filename, metrics=metrics
        )
        pickle.dump(self.genotype, open(geno_filename + ".p", "wb"))
        return img


class ReceiverAgent(BaseAgent):
    def __init__(self, run_folder, args, genotype=None, agent_id=0):
        if args.cell_type == "darts" and genotype is None:
            raise ValueError("Expected genotype in Receiver with option 'darts'")

        self.run_folder = run_folder
        self.agent_id = agent_id
        filename = "{}/receivers/receiver_{}.p".format(self.run_folder, agent_id)
        super().__init__(filename, args)
        self.genotype = genotype

        receiver = self.get_receiver(genotype)
        torch.save(receiver, filename)

    def get_receiver(self, genotype):
        if args.task == "shapes":
            receiver = ShapesReceiver(
                self.args.vocab_size,
                embedding_size=self.args.embedding_size,
                cell_type=self.args.cell_type,
                genotype=genotype,
            )
        if args.task == "obverter":
            receiver = ObverterReceiver(
                args.vocab_size,
                embedding_size=args.embedding_size,
                dataset_type=args.dataset_type,
                meta_vocab_size=args.meta_vocab_size,
                cell_type=args.cell_type,
                genotype=genotype,
            )
        return receiver

    def mutate(self, new_genotype):
        """
        Mutate model to new genotype
        """
        self.genotype = new_genotype

        receiver = self.get_receiver(new_genotype)
        torch.save(receiver, self.filename)

        self.initialize_loss_acc()
        self.age = 0

    def save_genotype(self, generation=0, metrics={}):
        geno_filename = "{}/receivers_genotype/receiver_{}_generation_{}".format(
            self.run_folder, self.agent_id, generation
        )
        img = get_genotype_image(
            self.genotype.recurrent, geno_filename, metrics=metrics
        )
        pickle.dump(self.genotype, open(geno_filename + ".p", "wb"))
        return img
