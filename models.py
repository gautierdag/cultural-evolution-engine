import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sender(nn.Module):
    def __init__(
        self,
        n_image_features,
        vocab_size,
        embedding_dim,
        hidden_size,
        batch_size,
        greedy=True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.greedy = greedy
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.aff_transform = nn.Linear(n_image_features, hidden_size)
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_dim), dtype=torch.float32)
        )
        self.linear_probs = nn.Linear(
            hidden_size, vocab_size
        )  # from a hidden state to the vocab

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.normal_(self.aff_transform.weight, 0, 0.1)
        nn.init.constant_(self.aff_transform.bias, 0)

        nn.init.constant_(self.linear_probs.weight, 0)
        nn.init.constant_(self.linear_probs.bias, 0)

        nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
        nn.init.orthogonal_(self.lstm_cell.weight_hh)
        nn.init.constant_(self.lstm_cell.bias_ih, val=0)

        # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
        # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
        nn.init.constant_(self.lstm_cell.bias_hh, val=0)
        nn.init.constant_(
            self.lstm_cell.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
        )

    def forward(self, t, start_token_idx, max_sentence_length, tau=1.2):

        if self.training:
            message = [
                torch.zeros(
                    (self.batch_size, self.vocab_size),
                    dtype=torch.float32,
                    device=device,
                )
            ]
            message[0][:, start_token_idx] = 1.0
        else:
            message = [
                torch.full(
                    (self.batch_size,),
                    fill_value=start_token_idx,
                    dtype=torch.int64,
                    device=device,
                )
            ]

        # h0, c0, w0
        h = self.aff_transform(t)  # batch_size, hidden_size
        c = torch.zeros([self.batch_size, self.hidden_size], device=device)

        for i in range(max_sentence_length):  # or sampled <S>, but this is batched
            emb = (
                torch.matmul(message[-1], self.embedding)
                if message[-1].dtype == torch.float32
                else self.embedding[message[-1]]
            )
            h, c = self.lstm_cell(emb, (h, c))

            p = F.softmax(self.linear_probs(h), dim=1)

            if self.training:
                rohc = RelaxedOneHotCategorical(tau, p)
                token = rohc.rsample()

                # Straight-through part
                token_hard = torch.zeros_like(token)
                token_hard.scatter_(-1, torch.argmax(token, dim=-1, keepdim=True), 1.0)
                token = (token_hard - token).detach() + token
            else:
                if self.greedy:
                    _, token = torch.max(p, -1)
                else:
                    token = Categorical(p).sample()

            message.append(token)

        return torch.stack(message[1:], dim=1)  # Skip the first <S>


class Receiver(nn.Module):
    def __init__(
        self, n_image_features, vocab_size, embedding_dim, hidden_size, batch_size
    ):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_dim), dtype=torch.float32)
        )
        self.aff_transform = nn.Linear(hidden_size, n_image_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.normal_(self.aff_transform.weight, 0, 0.1)
        nn.init.constant_(self.aff_transform.bias, 0)

        nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
        nn.init.orthogonal_(self.lstm_cell.weight_hh)
        nn.init.constant_(self.lstm_cell.bias_ih, val=0)
        # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
        # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
        nn.init.constant_(self.lstm_cell.bias_hh, val=0)
        nn.init.constant_(
            self.lstm_cell.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
        )

    def forward(self, m):
        # h0, c0
        h = torch.zeros([self.batch_size, self.hidden_size], device=device)
        c = torch.zeros([self.batch_size, self.hidden_size], device=device)

        # Need to change to batch dim second to iterate over tokens in message
        if len(m.shape) == 3:
            m = m.permute(1, 0, 2)
        else:
            m = m.permute(1, 0)

        for token in m:
            emb = (
                torch.matmul(token, self.embedding)
                if token.dtype == torch.float32
                else self.embedding[token]
            )
            h, c = self.lstm_cell(emb, (h, c))

        return self.aff_transform(h)


class Model(nn.Module):
    def __init__(
        self, n_image_features, vocab_size, embedding_dim, hidden_size, batch_size
    ):
        super().__init__()

        self.batch_size = batch_size
        self.sender = Sender(
            n_image_features, vocab_size, embedding_dim, hidden_size, batch_size
        )
        self.receiver = Receiver(
            n_image_features, vocab_size, embedding_dim, hidden_size, batch_size
        )

    def forward(self, target, distractors, start_token_idx, max_sentence_length):
        target = target.to(device)
        distractors = [d.to(device) for d in distractors]

        use_different_targets = len(target.shape) == 3
        assert (
            not use_different_targets or target.shape[1] == 2
        ), "This should only be two targets"

        if not use_different_targets:
            target_sender = target
            target_receiver = target
        else:
            target_sender = target[:, 0, :]
            target_receiver = target[:, 1, :]

        m = self.sender(target_sender, start_token_idx, max_sentence_length)

        r_transform = self.receiver(m)  # g(.)

        loss = 0

        target_receiver = target_receiver.view(self.batch_size, 1, -1)
        r_transform = r_transform.view(self.batch_size, -1, 1)

        target_score = torch.bmm(target_receiver, r_transform).squeeze()  # scalar

        distractors_scores = []

        for d in distractors:
            d = d.view(self.batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            distractors_scores.append(d_score)
            zero_tensor = torch.tensor(0.0).to(device)

            loss += torch.max(zero_tensor, 1.0 - target_score + d_score)

        # Calculate accuracy
        all_scores = torch.zeros((self.batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        for i, score in enumerate(distractors_scores):
            all_scores[:, i + 1] = score

        all_scores = torch.exp(all_scores)

        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        return torch.mean(loss), torch.mean(accuracy), m

