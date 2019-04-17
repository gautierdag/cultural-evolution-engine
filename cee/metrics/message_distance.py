import numpy as np
import itertools
from .rsa import one_hot


def message_distance(messages):
    """
    Args:
        message: N messages of length L from A agents, shape: N*A*L

    """
    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    encoded_messages = one_hot(messages).reshape(N, A, -1).astype(float)
    tot_dist = 0
    perfect_matches = 0
    for c in combinations:
        diff = np.sum(
            np.abs(encoded_messages[:, c[0], :] - encoded_messages[:, c[1], :]), axis=1
        )
        perfect_matches += np.count_nonzero(diff == 0)
        tot_dist += np.sum(diff)

    # average over number of number of combinations and examples
    tot_dist /= N * len(combinations)
    perfect_matches /= N * len(combinations)

    return tot_dist, perfect_matches
