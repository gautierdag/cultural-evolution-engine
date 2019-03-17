from utils import AverageMeter
import torch
import torch.nn as nn
from tqdm import tqdm


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
