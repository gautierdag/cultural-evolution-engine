import random
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
import os


class AverageMeter:
    def __init__(self):
        """
        Computes and stores the average and current value
        Taken from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_batch(model, batch, optimizer):
    """
    Train for single batch
    """
    model.train()
    optimizer.zero_grad()
    if len(batch) == 2:  # shapes
        target, distractors = batch
        loss, acc, _ = model(target, distractors)
    if len(batch) == 3:  # obverter task
        first_image, second_image, label = batch
        loss, acc, _ = model(first_image, second_image, label)
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def train_one_epoch(model, data, optimizer):
    """
    Train for a whole epoch
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for d in tqdm(data, total=len(data)):
        loss, acc = train_one_batch(model, d, optimizer)
        loss_meter.update(loss)
        acc_meter.update(acc)

    return loss_meter, acc_meter


def evaluate(model, data, return_softmax=False):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    entropy_meter = AverageMeter()
    hidden_sender, hidden_receiver = [], []
    messages, sentence_probabilities = [], []

    model.eval()
    for d in data:
        if len(d) == 2:  # shapes
            target, distractors = d
            loss, acc, msg, h_s, h_r, entropy, sent_p = model(target, distractors)

        if len(d) == 3:  # obverter task
            first_image, second_image, label = d
            loss, acc, msg, h_s, h_r, entropy, sent_p = model(
                first_image, second_image, label
            )

        loss_meter.update(loss.item())
        acc_meter.update(acc.item())
        entropy_meter.update(entropy.item())

        messages.append(msg)
        sentence_probabilities.append(sent_p)
        hidden_sender.append(h_s.detach().cpu().numpy())
        hidden_receiver.append(h_r.detach().cpu().numpy())

    hidden_sender = np.concatenate(hidden_sender)
    hidden_receiver = np.concatenate(hidden_receiver)

    if return_softmax:
        return (
            loss_meter,
            acc_meter,
            entropy_meter,
            torch.cat(messages, 0),
            torch.cat(sentence_probabilities, 0),
            hidden_sender,
            hidden_receiver,
        )
    else:
        return (
            loss_meter,
            acc_meter,
            entropy_meter,
            torch.cat(messages, 0),
            hidden_sender,
            hidden_receiver,
        )


class EarlyStopping:
    def __init__(self, mode="min", patience=10, threshold=5e-3, threshold_mode="rel"):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self.best = self.mode_worse

    def step(self, metrics):
        if self.is_converged:
            raise ValueError
        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.is_converged = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = float("inf")
        else:  # mode == 'max':
            self.mode_worse = -float("inf")

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)


def get_filename_from_baseline_params(params):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    if params.name:
        return params.name

    name = params.task
    name += "_{}".format(params.dataset_type)
    name += "_e_{}".format(params.embedding_size)
    name += "_h_{}".format(params.hidden_size)
    name += "_lr_{}".format(params.lr)
    name += "_max_len_{}".format(params.max_length)
    if params.task == "shapes":
        name += "_k_{}".format(params.k)
    name += "_vocab_{}".format(params.vocab_size)
    name += "_seed_{}".format(params.seed)
    name += "_btch_size_{}".format(params.batch_size)
    if params.single_model:
        name += "_single_model"
    if params.greedy:
        name += "_greedy"
    if params.debugging:
        name += "_debug"
    if params.sender_path or params.receiver_path:
        name += "_loaded_from_path"
    if params.obverter_setup:
        name = "obverter_setup_with_" + name
    return name


def get_filename_from_cee_params(params):
    """
    Generates a filename from cee params
    """
    name = "cee_{}_{}_pop_size_{}_cull_interval_{}_cull_rate_{}".format(
        params.task,
        params.dataset_type,
        params.population_size,
        params.culling_interval,
        params.culling_rate,
    )
    if params.single_pool:
        name += "_single_pool_"

    name += "_e{}".format(params.embedding_size)
    name += "_h{}".format(params.hidden_size)
    name += "_len{}".format(params.max_length)
    name += "_voc{}".format(params.vocab_size)
    if params.evolution:
        name = "evolution_" + name
        name += "_evo_mode_{}".format(params.evolution_mode)
    else:
        name += "_cull_mode_{}".format(params.culling_mode)
    if params.debugging:
        name += "_debug"

    return name


def seed_torch(seed=42):
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_folder_if_not_exists(folder_name):
    """
    Creates folder at folder name if folder does not exist
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
