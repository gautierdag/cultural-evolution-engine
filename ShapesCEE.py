from cee import BaseCEE
from cee.metrics import representation_similarity_analysis, language_entropy

from ShapesAgents import SenderAgent, ReceiverAgent
from model import Trainer, generate_genotype, mutate_genotype
from utils import create_folder_if_not_exists, train_one_batch, evaluate
import torch
from statistics import mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShapesCEE(BaseCEE):
    def __init__(self, params, run_folder="runs"):
        self.run_folder = run_folder
        super().__init__(params)

        if params.evolution:
            self.hall_of_shame = set()

    def initialize_population(self, params: dict):
        """
        Initializes params.population_size sender and receiver models
            Args:
                params (required): params obtained from argparse
        """
        create_folder_if_not_exists(self.run_folder + "/senders")
        create_folder_if_not_exists(self.run_folder + "/receivers")

        if params.evolution:
            create_folder_if_not_exists(self.run_folder + "/senders_genotype")
            create_folder_if_not_exists(self.run_folder + "/receivers_genotype")

        for i in range(params.population_size):
            sender_genotype = None
            receiver_genotype = None
            if params.evolution:
                sender_genotype = generate_genotype(num_nodes=params.init_nodes)
                receiver_genotype = generate_genotype(num_nodes=params.init_nodes)

            self.senders.append(
                SenderAgent(
                    self.run_folder, params, genotype=sender_genotype, agent_id=i
                )
            )
            self.receivers.append(
                ReceiverAgent(
                    self.run_folder, params, genotype=receiver_genotype, agent_id=i
                )
            )

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

    def evaluate_population(
        self, test_data, meta_data, features, advanced=True, max_senders=8
    ):
        """
        Evaluates language for population
            - need to get generated messages by all senders
            - since receivers don't talk - pick a random one
        Args:
            test_data: dataloader to evaluate against
            dataset (str, opt) from {"train", "valid", "test"}
            meta_data: encoded metadata for inputs
            features: features in test_data (in numpy array)
            advanced (bool, optional): whether to compute advanced metrics
            max_senders (int, optional): max number of senders to evaluate against
                                        evaluating over the entire set is costly
                                        so this approximation speeds it up
        """
        random.shuffle(self.senders)

        r = self.sample_population(receiver=True)

        total_loss, total_acc, total_entropy = 0, 0, 0
        total_l_entropy = 0  # language entropy
        rsa_sr, rsa_si, rsa_ri, topological_similarity = 0, 0, 0, 0
        num_unique_messages = 0

        messages = []
        for s in self.senders[:max_senders]:
            loss, acc, entropy, msgs, H_s, H_r = self.evaluate_pair(s, r, test_data)
            num_unique_messages += len(torch.unique(msgs, dim=0))
            if advanced:
                sr, si, ri, ts, l_entropy = self.get_message_metrics(
                    msgs, H_s, H_r, meta_data, features
                )
                rsa_sr += sr
                rsa_si += si
                rsa_ri += ri
                topological_similarity += ts
                total_l_entropy += l_entropy

            total_loss += loss
            total_acc += acc
            total_entropy += entropy

            messages.append(msgs)

        # @TODO:
        #   - implement language comparaison metric here (KL)
        #   - implement generalization error
        pop_size = len(self.senders)
        total_loss /= pop_size
        total_acc /= pop_size
        total_entropy /= pop_size
        rsa_sr /= pop_size
        rsa_si /= pop_size
        rsa_ri /= pop_size
        topological_similarity /= pop_size
        total_l_entropy /= pop_size
        num_unique_messages /= pop_size

        return (
            total_loss,
            total_acc,
            total_entropy,
            rsa_sr,
            rsa_si,
            rsa_ri,
            topological_similarity,
            total_l_entropy,
            num_unique_messages,
        )

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
        test_loss_meter, test_acc_meter, entropy_meter, test_messages, hidden_sender, hidden_receiver = evaluate(
            model, test_data
        )

        return (
            test_loss_meter.avg,
            test_acc_meter.avg,
            entropy_meter.avg,
            test_messages,
            hidden_sender,
            hidden_receiver,
        )

    @staticmethod
    def get_message_metrics(
        messages, hidden_sender, hidden_receiver, meta_data, img_features
    ):
        """
        Runs metrics on the generated messages (single set of messages)
        Args:
            messages: individual batch of generated messages
            meta_data: encoded meta_data
        """
        messages = messages.cpu().numpy()

        rsa_sr, rsa_si, rsa_ri, topological_similarity = representation_similarity_analysis(
            img_features, meta_data, messages, hidden_sender, hidden_receiver
        )

        # rsa = representation_similarity_analysis(messages, meta_data)
        l_entropy = language_entropy(messages)

        return rsa_sr, rsa_si, rsa_ri, topological_similarity, l_entropy

    def sort_agents(self, receiver=False, dynamic=True, k_shot=100):
        """
        dynamic - whether k_shot is based on minimum batch size
                  or on passed k_shot value
        K_shot - how many initial batches/training steps
                to take into account in the average loss
        """
        att = "receivers" if receiver else "senders"
        pop_size = len(getattr(self, att))

        k_shot = self.params.culling_interval
        # k_shot is minimum number of batches that have been seen by any agent
        # so as to make loss comparaisons fair - cap to 100 batches minimum
        if dynamic:
            for agent in getattr(self, att):
                k_shot = max(min(k_shot, agent.age), 100)

        agents = []
        values = []

        for a in range(pop_size):
            # check model has been run
            if getattr(self, att)[a].age < 1:
                avg_loss = 100.0  # high value for loss
            else:
                avg_loss = mean(getattr(self, att)[a].loss[:k_shot])

            agents.append(a)
            values.append(avg_loss)

        values, agents = zip(*sorted(zip(values, agents)))
        return list(agents), list(values)

    def mutate_population(self, receiver=False, culling_rate=0.2, mode="best"):
        """
        mutates Population according to culling rate and mode
        Args:
            culling_rate (float, optional): percentage of the population to replace
                                            default: 0.2
            mode (string, optional): argument for sampling
        """
        self.generation += 1

        att = "receivers" if receiver else "senders"
        pop_size = len(getattr(self, att))

        c = max(1, int(culling_rate * pop_size))

        print("Mutating {} agents from {} Population".format(c, att))
        # picks random networks to mutate
        if mode == "random":
            for _ in range(c):
                sampled_agent = self.sample_population(receiver=receiver, mode=mode)
                new_genotype = mutate_genotype(sampled_agent.genotype)
                sampled_agent.mutate(new_genotype, generation=self.generation)

        # mutates best agent to make child and place this child instead of worst agent
        if mode == "best":
            agents, _ = self.sort_agents(receiver=receiver)
            best_agent = getattr(self, att)[agents[0]]

            # replace worst c models with mutated version of best
            agents.reverse()  # resort from worst to best
            for w in agents[:c]:
                worst_agent = getattr(self, att)[agents[w]]
                # add hall of shame logic - note might get stuck in loop
                # self.hall_of_shame.add(worst_agent.genotype)
                new_genotype = mutate_genotype(
                    best_agent.genotype, hall_of_shame=self.hall_of_shame
                )
                worst_agent.mutate(new_genotype, generation=self.generation)

    def get_avg_age(self):
        """
        Returns average age
        """
        age = 0
        c = 0
        for r in self.receivers:
            age += r.age
            c += 1
        for s in self.senders:
            age += s.age
            c += 1
        return age / c

    def get_avg_convergence_at_step(self, step):
        """
        Returns average loss over the first training steps
        taken by similar agents
        """
        sender_agents, sender_losses = self.sort_agents(dynamic=False, k_shot=step)
        receiver_agents, receiver_losses = self.sort_agents(
            receiver=True, dynamic=False, k_shot=step
        )
        losses = sender_losses + sender_losses
        return mean(losses)
