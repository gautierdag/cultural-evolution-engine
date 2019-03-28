from cee import BaseCEE
from cee.metrics import representation_similarity_analysis, language_entropy

from ShapesAgents import SenderAgent, ReceiverAgent
from model import Trainer
from utils import create_folder_if_not_exists, train_one_batch, evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShapesCEE(BaseCEE):
    def __init__(self, params, run_folder="runs"):
        self.run_folder = run_folder
        super().__init__(params)

    def initialize_population(self, params: dict):
        """
        Initializes params.population_size sender and receiver models
            Args:
                params (required): params obtained from argparse
        """
        create_folder_if_not_exists(self.run_folder + "/senders")
        create_folder_if_not_exists(self.run_folder + "/receivers")
        for i in range(params.population_size):

            sender_filename = "{}/senders/sender_{}.p".format(self.run_folder, i)
            self.senders.append(SenderAgent(sender_filename, params))

            receiver_filename = "{}/receivers/receiver_{}.p".format(self.run_folder, i)
            self.receivers.append(ReceiverAgent(receiver_filename, params))

    def train_population(self, batch):

        sender = self.sample_population()
        receiver = self.sample_population(receiver=True)
        sender_model = sender.get_model()
        receiver_model = receiver.get_model()

        model = Trainer(sender_model, receiver_model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        loss, acc = train_one_batch(model, batch, optimizer)
        sender.update_loss_acc(loss, acc)
        receiver.update_loss_acc(loss, acc)

        # Update receiver and sender files with new state
        sender.save_model(model.sender)
        receiver.save_model(model.receiver)

    def evaluate_population(self, test_data, meta_data):
        """
        Evaluates language for population
            - need to get generated messages by all senders
            - since receivers don't talk - pick a random one
        Args:
            test_data: dataloader to evaluate against
            dataset (str, opt) from {"train", "valid", "test"}
        """
        r = self.sample_population(receiver=True)

        total_loss, total_acc, total_rsa, total_entropy = 0, 0, 0, 0

        messages = []
        for s in self.senders:
            loss, acc, msgs = self.evaluate_pair(s, r, test_data)
            rsa, entropy = self.get_message_metrics(msgs, meta_data)
            total_loss += loss
            total_acc += acc
            total_rsa += rsa
            total_entropy += entropy
            messages.append(msgs)

        # @TODO: implement language comparaison metric here (KL)

        total_loss /= len(self.senders)
        total_acc /= len(self.senders)
        total_rsa /= len(self.senders)
        total_entropy /= len(self.senders)

        return total_loss, total_acc, total_rsa, total_entropy

    @staticmethod
    def evaluate_pair(sender, receiver, test_data):
        """
        Evaluates pair of sender/receiver on test data and returns avg loss/acc
        and generated messages
        Args:
            sender_name (path): path of sender model
            receiver_name (path): path of receiver model
            test_data (dataloader): dataloader of data to evaluate on
        Returns:
            avg_loss (float): average loss over data
            avg_acc (float): average accuracy over data
            test_messages (tensor): generated messages from data
        """
        sender_model = sender.get_model()
        receiver_model = receiver.get_model()
        model = Trainer(sender_model, receiver_model)
        model.to(device)
        test_loss_meter, test_acc_meter, test_messages = evaluate(model, test_data)

        return test_loss_meter.avg, test_acc_meter.avg, test_messages

    @staticmethod
    def get_message_metrics(messages, meta_data):
        """
        Runs metrics on the generated messages (single set of messages)
        Args:
            messages: individual batch of generated messages
            meta_data: encoded meta_data
        """
        messages = messages.cpu().numpy()
        rsa = representation_similarity_analysis(messages, meta_data)
        l_entropy = language_entropy(messages)

        return rsa, l_entropy
