from graphviz import Digraph
from collections import namedtuple
import random

Genotype = namedtuple("Genotype", "recurrent concat")

PRIMITIVES = ["tanh", "relu", "sigmoid", "identity"]
STEPS = 8
CONCAT = 8

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


def mutate_genotype(genotype, edit_steps=1):
    """
    simplest mutation possible - edits a random single thing
    """
    for s in range(edit_steps):
        node = random.randint(0, len(genotype.recurrent) - 1)
        # mutate either connection or primitive
        mutation_type = random.randint(0, 1)

        # mutate primitive
        if mutation_type == 0:
            p = random.choice(PRIMITIVES)
            genotype.recurrent[node] = (p, genotype.recurrent[node][1])

        # mutate connection
        if mutation_type == 1:
            r = random.randint(0, node)
            genotype.recurrent[node] = (genotype.recurrent[node][0], r)

    return genotype


def plot_genotype(genotype, filename, view=False):
    g = Digraph(
        format="pdf",
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

    g.render(filename, view=view)


if __name__ == "__main__":
    g = generate_genotype()
    plot_genotype(g.recurrent, "experiments/recurrent1")
    g = mutate_genotype(g)
    plot_genotype(g.recurrent, "experiments/recurrent2")
