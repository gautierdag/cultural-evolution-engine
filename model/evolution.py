from graphviz import Digraph
from PIL import Image
import copy
import random
import numpy as np


# Original darts genotype
# Taken from: https://github.com/quark0/darts/blob/master/rnn/genotypes.py
class Genotype(object):
    def __init__(self, recurrent, concat):
        self.recurrent = recurrent
        self.concat = concat

    def __repr__(self):
        tmp = "Recurrent Gates: \n"
        for c in self.recurrent:
            tmp += "{}, {} \n".format(c[0], c[1])
        tmp += "concat: {}".format(self.concat)
        return tmp

    def __eq__(self, other):
        if len(self.recurrent) != len(other.recurrent):
            return False
        for i in range(len(self.recurrent)):
            if self.recurrent[i] != other.recurrent[i]:
                return False
        return True

    def __hash__(self):
        return hash(tuple(self.recurrent))


PRIMITIVES = ["tanh", "relu", "sigmoid", "identity"]
MAX_NODES = 8

DARTS = Genotype(
    recurrent=[
        ("sigmoid", 0),
        ("relu", 1),
        ("relu", 1),
        ("identity", 1),
        ("tanh", 2),
        ("sigmoid", 5),
        ("tanh", 3),
        ("relu", 5),
    ],
    concat=range(1, 9),
)

# Evolution Specific functions
# convert_genotype_to_bit_encoding?
# convert_genotype to tree ?
def generate_genotype(num_nodes=8):
    """
    Randomly generate a random genotype
    """
    recurrent = []
    for n in range(num_nodes):
        p = random.choice(PRIMITIVES)
        recurrent.append((p, random.randint(0, n)))

    return Genotype(recurrent=recurrent, concat=range(1, num_nodes + 1))


def mutate_genotype(genotype, edit_steps=1, allow_add_node=True):
    """
    simplest mutation possible - edits a random single thing
    if allow_add_node - might randomly add node in mutation if num_nodes < MAX_NODES
    """
    new_genotype = copy.deepcopy(genotype)
    number_of_nodes = len(new_genotype.recurrent)
    allow_add_node = allow_add_node and (number_of_nodes < MAX_NODES)
    for s in range(edit_steps):

        # mutate either connection or primitive
        mutation_type = random.randint(0, 1 + allow_add_node)
        node = random.randint(0, number_of_nodes - 1)
        # mutate primitive
        if mutation_type == 0:
            p = random.choice(PRIMITIVES)
            new_genotype.recurrent[node] = (p, new_genotype.recurrent[node][1])

        # mutate connection
        if mutation_type == 1:
            r = random.randint(0, node)
            new_genotype.recurrent[node] = (new_genotype.recurrent[node][0], r)

        # Add Node
        if mutation_type == 2:
            p = random.choice(PRIMITIVES)
            r = random.randint(0, number_of_nodes)
            new_genotype.recurrent.append((p, r))
            new_genotype.concat = range(1, number_of_nodes + 2)

    return new_genotype


def load_image(infilename: str):
    """
    helper for loading image from file to numpy
    """
    img = Image.open(infilename)
    data = np.array(img).astype(float) / 255
    return data


def get_genotype_image(genotype, filename, view=False, metrics={}):
    g = Digraph(
        format="jpeg",
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    g.node("x_{t}", fillcolor="darkseagreen2")
    g.node("h_{t-1}", fillcolor="darkseagreen2")
    g.node("0", fillcolor="lightblue")
    g.edge("x_{t}", "0", fillcolor="gray")
    g.edge("h_{t-1}", "0", fillcolor="gray")
    steps = len(genotype)

    for i in range(1, steps + 1):
        g.node(str(i), fillcolor="lightblue")

    for i, (op, j) in enumerate(genotype):
        g.edge(str(j), str(i + 1), label=op, fillcolor="gray")

    g.node("h_{t}", fillcolor="palegoldenrod")
    for i in range(1, steps + 1):
        g.edge(str(i), "h_{t}", fillcolor="gray")

    atts = ""
    for k in metrics:
        atts += "{}: {} \n".format(k, round(metrics[k], 4))
    g.attr(label=atts)

    g.render(filename, view=view, format="jpeg")
    return load_image(filename + ".jpeg")


if __name__ == "__main__":
    g = generate_genotype(num_nodes=1)
    # get_genotype_image(g.recurrent, "experiments/recurrent1")
    print(g)
    h = mutate_genotype(g)
    print(h)
    i = mutate_genotype(g)
    print(i)
    # get_genotype_image(g.recurrent, "experiments/recurrent2")
