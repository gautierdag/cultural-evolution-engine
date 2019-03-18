import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class Receiver(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=256, hidden_size=512):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
#         self.embedding = nn.Parameter(
#             torch.empty((vocab_size, embedding_dim), dtype=torch.float32)
#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.normal_(self.embedding, 0.0, 0.1)

#         nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
#         nn.init.orthogonal_(self.lstm_cell.weight_hh)
#         nn.init.constant_(self.lstm_cell.bias_ih, val=0)
#         nn.init.constant_(self.lstm_cell.bias_hh, val=0)
#         nn.init.constant_(
#             self.lstm_cell.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
#         )

#     def forward(self, messages):
#         batch_size = messages.shape[0]
#         # h0, c0
#         h = torch.zeros([batch_size, self.hidden_size], device=device)
#         c = torch.zeros([batch_size, self.hidden_size], device=device)

#         # Need to change to batch dim second to iterate over tokens in message
#         if len(messages.shape) == 3:
#             messages = messages.permute(1, 0, 2)
#         else:
#             messages = messages.permute(1, 0)

#         for token in messages:
#             emb = (
#                 torch.matmul(token, self.embedding)
#                 if token.dtype == torch.float32
#                 else self.embedding[token]
#             )
#             h, c = self.lstm_cell(emb, (h, c))

#         return h


class Receiver(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=512):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1)
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

    def forward(self, messages, seq_lengths):
        batch_size = messages.shape[0]

        emb = (
            torch.matmul(messages, self.embedding)
            if messages.dtype == torch.float32
            else self.embedding[messages]
        )
        emb = emb.permute(1, 0, 2)

        _, (h_last, _) = self.lstm(emb)

        return h_last
