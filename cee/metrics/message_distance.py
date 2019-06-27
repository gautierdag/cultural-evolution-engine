import numpy as np
import itertools
import scipy.spatial
import scipy.stats
from sklearn.metrics import jaccard_score
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


def jaccard_similarity(messages, samples=200):
    """
    Averages average jaccard similarity between all pairs of agents.
    Args:
        messages (ndarray, ints): N messages of length L from A agents, shape: N*A*L
    Returns:
        score (float): average jaccard similarity between all pairs of agents.
    """
    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    score = 0.0
    for c in combinations:
        for _ in range(samples):
            s = np.random.randint(N)
            score += jaccard_score(
                messages[s, c[0], :], messages[s, c[1], :], average="macro"
            )

    # average over number of combinations
    score /= len(combinations) * samples

    return score


def kl_divergence(messages, eps=1e-6, samples=200):
    """
    Aproximates average KL divergence between all pairs of agents.
    Args:
        messages (ndarray, ints): N probability of messages length V (vocab size) from A agents, shape: N*A*V
    Returns:
        score (float): average pair-wise KL divergence
    """
    N, A = messages.shape[0], messages.shape[1]

    vocab_size = messages.max() + 1

    score = 0.0
    count = 1
    for i in range(A):
        for j in range(A):
            if j == i:
                continue
            else:
                for _ in range(samples):
                    s = np.random.randint(N)
                    score += scipy.stats.entropy(
                        messages[s, i, :] + eps, messages[s, j, :] + eps
                    )
                    count += 1
    # average over number of combinations
    score /= count

    return score
