import random


class BaseCEE(object):
    def __init__(self, params):
        self.senders = []
        self.receivers = []
        self.params = params
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
        pop_size = len(self.receivers) if receiver else len(self.senders)
        if mode == "random":
            r = random.randrange(0, pop_size)
        else:
            raise ValueError("mode={} undefined for sampling population".format(mode))

        return self.receivers[r] if receiver else self.senders[r]

    def cull_population(self, receiver=False, culling_rate=0.2, mode="random"):
        """
        Culls Population according to culling rate and mode
        Args:
            culling_rate (float, optional): percentage of the population to cull
                                            default: 0.2
            mode (string, optional): argument for sampling
        """
        pop_size = len(self.receivers) if receiver else len(self.senders)
        c = max(1, int(culling_rate * pop_size))
        for _ in range(c):
            sampled_model = self.sample_population(receiver=receiver, mode=mode)
            sampled_model.cull()
