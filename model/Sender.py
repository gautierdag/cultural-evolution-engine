import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from .gumbel import gumbel_softmax
from .DARTSCell import DARTSCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sender(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_len,
        sos_id,
        eos_id=None,
        embedding_size=256,
        hidden_size=512,
        greedy=False,
        cell_type="lstm",
        genotype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_len = output_len
        self.sos_id = sos_id

        if eos_id is None:
            self.eos_id = sos_id
        else:
            self.eos_id = eos_id

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.greedy = greedy

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        elif cell_type == "darts":
            self.rnn = DARTSCell(embedding_size, hidden_size, genotype)

        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )
        self.linear_out = nn.Linear(
            hidden_size, vocab_size
        )  # from a hidden state to the vocab

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
            # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def _init_state(self, hidden_state, rnn_type):
        """
            Handles the initialization of the first hidden state of the decoder.
            Hidden state + cell state in the case of an LSTM cell or
            only hidden state in the case of a GRU cell.
            Args:
                hidden_state (torch.tensor): The state to initialize the decoding with.
                rnn_type (type): Type of the rnn cell.
            Returns:
                state: (h, c) if LSTM cell, h if GRU cell
                batch_size: Based on the given hidden_state if not None, 1 otherwise
        """

        # h0
        if hidden_state is None:
            batch_size = 1
            h = torch.zeros([batch_size, self.hidden_size], device=device)
        else:
            batch_size = hidden_state.shape[0]
            h = hidden_state  # batch_size, hidden_size

        # c0
        if rnn_type is nn.LSTMCell:
            c = torch.zeros([batch_size, self.hidden_size], device=device)
            state = (h, c)
        else:
            state = h

        return state, batch_size

    def _calculate_seq_len(
        self, seq_lengths, token, initial_length, seq_pos, n_sos_symbols, is_discrete
    ):
        """
            Calculates the lengths of each sequence in the batch in-place.
            The length goes from the start of the sequece up until the eos_id is predicted.
            If it is not predicted, then the length is output_len + n_sos_symbols.
            Args:
                seq_lengths (torch.tensor): To keep track of the sequence lengths.
                token (torch.tensor): Batch of predicted tokens at this timestep.
                initial_length (int): The max possible sequence length (output_len + n_sos_symbols).
                seq_pos (int): The current timestep.
                n_sos_symbols (int): Number of sos symbols at the beginning of the sequence.
                is_discrete (bool): True if Gumbel Softmax is used, False otherwise.
        """
        if is_discrete:
            mask = token == self.eos_id
        else:
            max_predicted, vocab_index = torch.max(token, dim=1)
            mask = (vocab_index == self.eos_id) * (max_predicted == 1.0)
        mask *= seq_lengths == initial_length
        seq_lengths[mask.nonzero()] = seq_pos + n_sos_symbols

    def forward(self, tau, hidden_state=None):
        """
        Performs a forward pass. If training, use Gumbel Softmax (hard) for sampling, else use
        discrete sampling.
        Hidden state here represents the encoded image - initializes the RNN from it.
        """

        state, batch_size = self._init_state(hidden_state, type(self.rnn))

        # Init output
        if self.training:
            output = [
                torch.zeros(
                    (batch_size, self.vocab_size), dtype=torch.float32, device=device
                )
            ]
            output[0][:, self.sos_id] = 1.0
        else:
            output = [
                torch.full(
                    (batch_size,),
                    fill_value=self.sos_id,
                    dtype=torch.int64,
                    device=device,
                )
            ]

        # Keep track of sequence lengths
        n_sos_symbols = 1
        initial_length = self.output_len + n_sos_symbols
        seq_lengths = (
            torch.ones([batch_size], dtype=torch.int64, device=device) * initial_length
        )

        for i in range(self.output_len):
            if self.training:
                emb = torch.matmul(output[-1], self.embedding)
            else:
                emb = self.embedding[output[-1]]

            state = self.rnn(emb, state)

            if type(self.rnn) is nn.LSTMCell:
                h, c = state
            else:
                h = state

            p = F.softmax(self.linear_out(h), dim=1)

            if self.training:
                token = gumbel_softmax(p, tau, hard=True)
            else:
                if self.greedy:
                    _, token = torch.max(p, -1)
                else:
                    token = Categorical(p).sample()

                if batch_size == 1:
                    token = token.unsqueeze(0)

            output.append(token)

            self._calculate_seq_len(
                seq_lengths,
                token,
                initial_length,
                seq_pos=i + 1,
                n_sos_symbols=n_sos_symbols,
                is_discrete=not self.training,
            )

        return (torch.stack(output, dim=1), seq_lengths, emb)
