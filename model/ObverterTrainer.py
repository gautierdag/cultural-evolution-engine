import numpy as np
import torch
import torch.nn as nn
from .visual_module import CNN


class ObverterTrainer(nn.Module):
    def __init__(self, sender, receiver, extract_features=False, device=None):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

        self.extract_features = extract_features
        if extract_features:
            self.visual_module = CNN(sender.hidden_size)

        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        batch_size, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=self.device).expand(
            len(seq_lengths), max_len
        ) < seq_lengths.unsqueeze(1)

        if self.training:
            mask = mask.type(dtype=messages.dtype)
            messages = messages * mask.unsqueeze(2)
            # give full probability (1) to eos tag (used as padding in this case)
            messages[:, :, self.sender.eos_id] += (mask == 0).type(dtype=messages.dtype)
        else:
            # fill in the rest of message with eos
            messages = messages.masked_fill_(mask == 0, self.sender.eos_id)

        return messages

    def forward(self, first_image, second_image, label):
        batch_size = first_image.shape[0]

        first_image = first_image.to(self.device)
        second_image = second_image.to(self.device)

        if self.extract_features:
            first_image = self.visual_module(first_image)
            second_image = self.visual_module(second_image)

        label = label.to(self.device)

        messages, lengths, entropy, h_s, sentence_prob = self.sender(
            first_image, device=self.device
        )
        messages = self._pad(messages, lengths)
        prediction, h_r = self.receiver(
            second_image, messages=messages, device=self.device
        )

        loss = self.loss(prediction, label)

        # Calculate accuracy
        accuracy = (
            torch.sum(torch.argmax(prediction, dim=1) == label).type(torch.float)
            / batch_size
        )

        if self.training:
            return loss, accuracy, messages
        else:
            return loss, accuracy, messages, h_s, h_r, entropy, sentence_prob

