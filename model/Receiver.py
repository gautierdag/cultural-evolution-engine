import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Receiver(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=512):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True)
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, val=0)
        nn.init.constant_(self.lstm.bias_hh_l0, val=0)
        nn.init.constant_(
            self.lstm.bias_hh_l0[self.hidden_size : 2 * self.hidden_size], val=1
        )

    def forward(self, messages):
        batch_size = messages.shape[0]

        emb = (
            torch.matmul(messages, self.embedding)
            if self.training
            else self.embedding[messages]
        )

        _, (h_last, _) = self.lstm(emb)

        return h_last, emb
