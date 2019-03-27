import torch
import numpy as np
import scipy.spatial
import scipy.stats


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def encode_messages(messages):
    encoded_messages = messages.copy()
    eos_token = encoded_messages.max()
    # pad
    for m in range(encoded_messages.shape[0]):
        for t in range(1, encoded_messages.shape[1]):
            if encoded_messages[m][t] == eos_token:
                encoded_messages[m, t:] = 0
    # remove eos
    encoded_messages = encoded_messages[:, 1:]

    # one hot
    encoded_messages = one_hot(encoded_messages)

    return encoded_messages.reshape(encoded_messages.shape[0], -1)


def representation_similarity_analysis(generated_messages, test_set, samples=5000):
    """
    Args:
        generated_messages: generated messages output from eval on test
        test_set: encoded test set metadata info describing the image
    """
    # encode messages by taking padding into account and transforming to one hot
    messages = encode_messages(generated_messages)
    # this is needed since some samples might have been dropped during training to maintain batch_size
    test_set = test_set[: len(messages)]
    assert test_set.shape[0] == messages.shape[0]

    sim_reals = np.zeros(samples)
    sim_msgs = np.zeros(samples)
    for i in range(samples):
        rnd = np.random.choice(len(test_set), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]
        sim_reals[i] = scipy.spatial.distance.cosine(messages[s1], messages[s2])
        sim_msgs[i] = scipy.spatial.distance.cosine(test_set[s1], test_set[s2])

    rsa = scipy.stats.pearsonr(sim_reals, sim_msgs)[0]

    return rsa
