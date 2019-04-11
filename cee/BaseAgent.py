import torch


class BaseAgent(object):
    def __init__(self, filename, args={}):
        self.filename = filename
        self.age = 0

        # Arguments used to generate agent
        self.args = args
        self.initialize_loss_acc()

    def increment_age(self):
        self.age += 1

    def initialize_loss_acc(self):
        self.loss = []
        self.acc = []

    def update_loss_acc(self, loss: float, acc: float):
        self.loss += [loss]
        self.acc += [acc]
        self.increment_age()

    def cull(self):
        """
        Reinitialize the weights of a single model
        """
        model = torch.load(self.filename)
        model.reset_parameters()
        torch.save(model, self.filename)

        self.age = 0
        self.initialize_loss_acc()

    def get_model(self):
        model = torch.load(self.filename)
        return model

    def save_model(self, model):
        torch.save(model, self.filename)
