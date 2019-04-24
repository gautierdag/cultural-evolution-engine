import itertools
from sklearn.metrics import jaccard_similarity_score


def jaccard_similarity(messages):
    """
    Args:
        message: N messages of length L from A agents, shape: N*A*L

    """
    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    score = 0.0
    for c in combinations:
        score += jaccard_similarity_score(
            encoded_messages[:, c[0], :], encoded_messages[:, c[1], :]
        )

    # average over number of combinations
    score /= len(combinations)

    return score
