import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial


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


def train_one_epoch(model, data, optimizer, start_token_idx, max_length):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()
    for d in tqdm(data, total=len(data)):
        optimizer.zero_grad()
        target, distractors = d
        loss, acc, _ = model(target, distractors, start_token_idx, max_length)
        loss_meter.update(loss.item())
        acc_meter.update(acc.item())
        loss.backward()
        optimizer.step()

    return loss_meter, acc_meter


def evaluate(model, data, start_token_idx, max_length):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    messages = []

    model.eval()
    for d in data:
        target, distractors = d
        loss, acc, msg = model(target, distractors, start_token_idx, max_length)
        loss_meter.update(loss.item())
        acc_meter.update(acc.item())
        messages.append(msg)

    return loss_meter, acc_meter, torch.cat(messages, 0)


class EarlyStopping:
    def __init__(self, mode="min", patience=30, threshold=5e-3, threshold_mode="rel"):
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
