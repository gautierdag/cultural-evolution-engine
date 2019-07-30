import torch
import random
import pickle
import copy
from statistics import mean
import numpy as np
import scipy

from cee import BaseCEE
from cee.metrics import (
    representation_similarity_analysis,
    language_entropy,
    message_distance,
    kl_divergence,
    jaccard_similarity,
)

from EvolutionAgents import SenderAgent, ReceiverAgent, SingleAgent
from model import ShapesTrainer, ObverterTrainer, generate_genotype, mutate_genotype
from utils import create_folder_if_not_exists, train_one_batch, evaluate


class EvolutionCEE(BaseCEE):
    def __init__(self, params, run_folder="runs", device=None):
        self.run_folder = run_folder
        super().__init__(params)

        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self):
        pickle.dump(self, open(self.run_folder + "/cee.p", "wb"))

    def initialize_population(self, params: dict):
        """
        Initializes params.population_size sender and receiver models
            Args:
                params (required): params obtained from argparse
        """
        if params.save_example_batch:
            create_folder_if_not_exists(self.run_folder + "/messages")

        if params.single_pool:
            create_folder_if_not_exists(self.run_folder + "/agents")
            if params.evolution:
                create_folder_if_not_exists(self.run_folder + "/agents_genotype")
        else:
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

            if params.single_pool:
                self.agents.append(
                    SingleAgent(
                        self.run_folder, params, genotype=sender_genotype, agent_id=i
                    )
                )
            else:
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
        if self.params.single_pool:
            sender, receiver = self.sample_agents_pair()
        else:
            sender = self.sample_population()
            receiver = self.sample_population(receiver=True)

        sender_model = sender.get_model()
        receiver_model = receiver.get_model()

        model = self.get_trainer(sender_model, receiver_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr)

        loss, acc = train_one_batch(model, batch, optimizer)
        sender.update_loss_acc(loss, acc)
        receiver.update_loss_acc(loss, acc)

        # Update receiver and sender files with new state
        sender.save_model(model.sender)
        receiver.save_model(model.receiver)

        self.iteration += 1

    def save_messages(self, messages, sender, i):
        filename = "{}/messages/message_from_{}_at_{}".format(
            self.run_folder, sender.agent_id, i
        )
        messages = messages.cpu().numpy()
        pickle.dump(messages, open(filename, "wb"))

    def evaluate_population(
        self,
        test_data,
        meta_data,
        features,
        advanced=False,
        max_senders=16,
        save_example_batch=False,
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
        if self.params.single_pool:
            random.shuffle(self.agents)
            sender_pop = self.agents[:max_senders]
        else:
            random.shuffle(self.senders)
            sender_pop = self.senders[:max_senders]

        r = self.sample_population(receiver=True)

        metrics = {
            "loss": 0,
            "acc": 0,
            "entropy": 0,
            "l_entropy": 0,  # language entropy
            "rsa_sr": 0,
            "rsa_si": 0,
            "rsa_ri": 0,
            "rsa_sm": 0,
            "pseudo_tre": 0,
            "topological_similarity": 0,
            "num_unique_messages": 0,
            "kl_divergence": 0,
        }

        messages = []
        sentence_probabilities = []
        for s in sender_pop:
            loss, acc, entropy, msgs, sent_ps, H_s, H_r = self.evaluate_pair(
                s, r, test_data
            )
            metrics["num_unique_messages"] += len(torch.unique(msgs, dim=0))

            if save_example_batch:
                self.save_messages(msgs, s, save_example_batch)

            if advanced:
                sr, si, ri, sm, ts, pt, l_entropy = self.get_message_metrics(
                    msgs, H_s, H_r, meta_data, features
                )

                metrics["rsa_sr"] += sr
                metrics["rsa_si"] += si
                metrics["rsa_ri"] += ri
                metrics["rsa_sm"] += sm
                metrics["topological_similarity"] += ts
                metrics["pseudo_tre"] += pt
                metrics["l_entropy"] += l_entropy

            metrics["loss"] += loss
            metrics["acc"] += acc
            metrics["entropy"] += entropy

            messages.append(msgs)
            sentence_probabilities.append(sent_ps)

        pop_size = max_senders
        for metric in metrics:
            metrics[metric] /= pop_size

        # language comparaison metric
        avg_message_dist, avg_matches = message_distance(
            torch.stack(messages, dim=1).cpu().numpy()
        )

        js = jaccard_similarity(torch.stack(messages, dim=1).cpu().numpy())

        kl_dist = kl_divergence(
            torch.stack(sentence_probabilities, dim=1).cpu().numpy()
        )

        metrics["jaccard_similarity"] = js
        metrics["kl_divergence"] = kl_dist

        metrics["avg_message_dist"] = avg_message_dist
        metrics["avg_matches"] = avg_matches

        return metrics

    def evaluate_pair(self, sender, receiver, test_data):
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

        model = self.get_trainer(sender_model, receiver_model)
        test_loss_meter, test_acc_meter, entropy_meter, test_messages, sentence_probabilities, hidden_sender, hidden_receiver = evaluate(
            model, test_data, return_softmax=True
        )

        return (
            test_loss_meter.avg,
            test_acc_meter.avg,
            entropy_meter.avg,
            test_messages,
            sentence_probabilities,
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

        rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity, pseudo_tre = representation_similarity_analysis(
            img_features, meta_data, messages, hidden_sender, hidden_receiver, tre=True
        )

        # rsa = representation_similarity_analysis(messages, meta_data)
        l_entropy = language_entropy(messages)

        return (
            rsa_sr,
            rsa_si,
            rsa_ri,
            rsa_sm,
            topological_similarity,
            pseudo_tre,
            l_entropy,
        )

    def get_convergence(self, att, dynamic=True, k_shot=100):
        pop_size = len(getattr(self, att))

        if dynamic:
            # k_shot is minimum number of batches that have been seen by any agent
            k_shot = self.params.culling_interval
            for agent in getattr(self, att):
                # so as to make loss comparaisons fair - cap to 100 batches minimum
                k_shot = max(min(k_shot, agent.age), 100)

        agents = []
        values = []

        for a in range(pop_size):
            # check model has been run
            if getattr(self, att)[a].age < 1 or (
                not dynamic and getattr(self, att)[a].age < k_shot
            ):
                avg_loss = 100.0  # high value for loss
            else:
                avg_loss = mean(getattr(self, att)[a].loss[:k_shot])

            # store the latest convergence for each agent
            getattr(self, att)[a].convergence = avg_loss

            agents.append(a)
            values.append(avg_loss)

        return values, agents

    def sort_agents(self, receiver=False, dynamic=True, k_shot=100):
        """
        Sorts agents according to convergence (see get_convergence)
        dynamic - whether k_shot is based on minimum batch size
                  or on passed k_shot value
        K_shot - how many initial batches/training steps
                to take into account in the average loss
        """
        if self.params.single_pool:
            att = "agents"
        else:
            att = "receivers" if receiver else "senders"

        values, agents = self.get_convergence(att, dynamic=dynamic, k_shot=k_shot)

        values, agents = zip(*sorted(zip(values, agents)))
        return list(agents), list(values)

    def save_best_agent(self, att, agent):
        agent_filename = "{}/best_{}_at_{}".format(
            self.run_folder, att[:-1], self.iteration - 1
        )
        pickle.dump(agent, open(agent_filename + ".p", "wb"))

    def mutate_population(self, receiver=False, culling_rate=0.2, mode="best"):
        """
        mutates Population according to culling rate and mode
        Args:
            culling_rate (float, optional): percentage of the population to replace
                                            default: 0.2
            mode (string, optional): argument for sampling {best, greedy}
        """

        if self.params.single_pool:
            att = "agents"
        else:
            att = "receivers" if receiver else "senders"

        pop_size = len(getattr(self, att))

        c = max(1, int(culling_rate * pop_size))

        print("Mutating {} agents from {} Population".format(c, att))

        # mutates best agent to make child and place this child instead of worst agent
        if mode == "best":
            agents, _ = self.sort_agents(receiver=receiver)
            best_agent = getattr(self, att)[agents[0]]
            self.save_best_agent(att, best_agent)

            # replace worst c models with mutated version of best
            agents.reverse()  # resort from worst to best
            for w in agents[:c]:
                worst_agent = getattr(self, att)[w]
                new_genotype = mutate_genotype(best_agent.genotype)
                worst_agent.mutate(new_genotype)

        if mode == "greedy":
            agents, values = self.sort_agents(receiver=receiver)
            best_agent = getattr(self, att)[agents[0]]
            self.save_best_agent(att, best_agent)

            # deep copy in case best agent is selected to be culled
            best_geno = copy.deepcopy(best_agent.genotype)

            # replace sampled worst c models with mutated version of best
            p = scipy.special.softmax(np.array(values))
            selected_agents = np.random.choice(agents, c, p=p, replace=False)
            for w in selected_agents:
                worst_agent = getattr(self, att)[w]
                new_genotype = mutate_genotype(best_geno)
                worst_agent.mutate(new_genotype)

    def get_avg_age(self):
        """
        Returns average age
        """
        age = 0
        c = 0
        if self.params.single_pool:
            for r in self.agents:
                age += r.age
                c += 1
        else:
            for r in self.receivers:
                age += r.age
                c += 1
            for s in self.senders:
                age += s.age
                c += 1
        return age / c

    def get_avg_convergence_at_step(self, step=10, dynamic=False):
        """
        Returns average loss over the first training steps
        taken by similar agents
        """
        sender_agents, sender_losses = self.sort_agents(dynamic=dynamic, k_shot=step)
        receiver_agents, receiver_losses = self.sort_agents(
            receiver=True, dynamic=False, k_shot=step
        )
        losses = sender_losses + receiver_losses
        return mean(losses)

    def save_genotypes_to_writer(self, writer, receiver=False):
        if self.params.single_pool:
            att = "agents"
        else:
            att = "receivers" if receiver else "senders"

        self.sort_agents(receiver=receiver)
        for a in getattr(self, att):
            m = {
                "agent id": a.agent_id,
                "loss": a.loss[-1],
                "convergence": a.convergence,
                "acc": a.acc[-1],
                "age": a.age,
            }
            img = a.save_genotype(generation=self.generation, metrics=m)
            writer.add_image(
                "{}{}".format(att, a.agent_id),
                img,
                global_step=self.generation,
                dataformats="HWC",
            )

    def get_trainer(self, sender, receiver):
        if self.params.task == "shapes":
            model = ShapesTrainer(sender, receiver, device=self.device)
        elif self.params.task == "obverter":
            model = ObverterTrainer(sender, receiver, device=self.device)
        else:
            raise ValueError("Incorrect task parameter")

        model.to(self.device)
        return model
