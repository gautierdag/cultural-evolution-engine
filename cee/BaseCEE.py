import random
import numpy as np
import scipy


class BaseCEE(object):
    def __init__(self, params):
        self.senders = []
        self.receivers = []
        self.agents = []  # case where single pool of agents
        self.params = params
        self.generation = 0
        self.iteration = 0
        self.initialize_population(params)

    def initialize_population(self, params: dict):
        raise NotImplementedError("Initialize population needs to be implemented")

    def train_population(self, batch):
        raise NotImplementedError("Train population needs to be implemented")

    def evaluate_population(self):
        raise NotImplementedError("Evaluate population needs to be implemented")

    def sample_population(self, receiver=False, mode: str = "random"):
        """
        population (dict): population dictionary containing a single population.
                            keys should be filenames and values attribute to do
                            selection on
        mode: pick from {'random'}
            - random: uniformly sample from population to cull ()
        """
        if self.params.single_pool:
            att = "agents"
        else:
            att = "receivers" if receiver else "senders"

        pop_size = len(getattr(self, att))

        if mode == "random":
            r = random.randrange(0, pop_size)
        else:
            raise ValueError("mode={} undefined for sampling population".format(mode))

        return getattr(self, att)[r]

    def sample_agents_pair(self, mode: str = "random"):
        """
        samples two agents from agent pool with no replacement
        mode: pick from {'random'}
            - random: uniformly sample from population to cull ()
        """
        pop_size = len(self.agents)
        if mode == "random":
            rnd = np.random.choice(pop_size, 2, replace=False)
            s1, s2 = rnd[0], rnd[1]
        else:
            raise ValueError("mode={} undefined for sampling population".format(mode))
        return (self.agents[s1], self.agents[s2])

    def sort_agents(self, receiver=False):
        raise NotImplementedError("sort_agents not implemented")

    def cull_population(self, receiver=False, culling_rate=0.2, mode="random"):
        """
        Culls Population according to culling rate and mode
        Args:
            culling_rate (float, optional): percentage of the population to cull
                                            default: 0.2
            mode (string, optional): argument for sampling
        """
        self.generation += 1

        if self.params.single_pool:
            att = "agents"
        else:
            att = "receivers" if receiver else "senders"

        pop_size = len(getattr(self, att))
        c = max(1, int(culling_rate * pop_size))

        if mode == "random":
            for _ in range(c):
                sampled_model = self.sample_population(receiver=receiver, mode=mode)
                sampled_model.cull()

        # sort by best converging
        if mode == "best":
            agents, _ = self.sort_agents(receiver=receiver)
            # cull worst c models
            agents.reverse()  # resort from worst to best
            for w in agents[:c]:
                worst_agent = getattr(self, att)[w]
                worst_agent.cull()

        if mode == "greedy":
            agents, values = self.sort_agents(receiver=receiver)
            p = scipy.special.softmax(np.array(values))
            selected_agents = np.random.choice(agents, c, p=p, replace=False)
            for w in selected_agents:
                worst_agent = getattr(self, att)[w]
                worst_agent.cull()

        # order by age
        if mode == "age":
            agents = []
            ages = []
            for a in range(pop_size):
                ages.append(getattr(self, att)[a].age)
                agents.append(a)
            # sort from oldest to newest
            ages, agents = zip(*sorted(zip(ages, agents), reverse=True))
            agents = list(agents)
            for w in agents[:c]:
                worst_agent = getattr(self, att)[w]
                worst_agent.cull()
