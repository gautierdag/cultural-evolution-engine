import numpy as np
import scipy.stats


def language_entropy(generated_messages):
    """
    Pads messages with 0 after a eos and ignores sos (start token)
    Then runs entropy on the full sequence
    Args:
        generated_messages: generated messages output from eval on test
    """
    padded_messages = np.zeros(
        (generated_messages.shape[0], generated_messages.shape[1] - 1)
    )
    for m in range(generated_messages.shape[0]):
        run_entropy_on_full = True
        for t in range(1, generated_messages.shape[1]):
            if messages[m][t] == eos_token:
                padded_messages[m, : t - 1] = messages[m, 1:t]
                run_entropy_on_full = False
        if run_entropy_on_full:
            padded_messages[m] = messages[m, 1:]

    y = np.bincount(padded_messages.flatten().astype(np.int))
    return scipy.stats.entropy(y)
