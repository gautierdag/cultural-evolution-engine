import torch
import torch.nn as nn
from tqdm import tqdm


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
