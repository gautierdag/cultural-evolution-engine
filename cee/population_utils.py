import torch
import random


def sample_population(population, mode="random"):
    """
    population (dict): population dictionary containing a single population.
                           keys should be filenames and values attribute to do
                           selection on
    mode: pick from {'random'}
        - random: uniformly sample from population to cull ()
    """
    if mode == "random":
        r = random.randrange(0, len(population))
        return list(population.keys())[r]
    else:
        raise ValueError("mode={} undefined for sampling population".format(mode))


def cull_model(model_filepath):
    """
    Reinitialize the weights of a single model
    """
    model = torch.load(model_filepath)
    model.reset_parameters()
    torch.save(model, model_filepath)


def cull_population(population, culling_rate=0.2, mode="random"):
    """
    Culls Population according to culling rate and mode
    Args:
        population (dict): population dictionary containing a single population.
                           keys should be filenames and values attribute to do
                           selection on
        culling_rate (float, optional): percentage of the population to cull
                                        default: 0.2
        mode (string, optional): argument for sampling
    """
    c = max(1, int(culling_rate * len(population)))
    print("Culling {} models from sender and receiver populations".format(c))
    for _ in range(c):
        sampled_model = sample_population(population, mode=mode)
        cull_model(sampled_model)


if __name__ == "__main__":
    a = {"1": 2, "2": 3, "3": 1}
    b = sample_population(a)
