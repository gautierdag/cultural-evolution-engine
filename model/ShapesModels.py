import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from .gumbel import gumbel_softmax
from .DARTSCell import DARTSCell


class ShapesMetaVisualModule(nn.Module):
    def __init__(
        self,
        features_dim=512,
        meta_dim=15,
        hidden_size=512,
        dataset_type="meta",
        sender=True,
    ):
        super(ShapesMetaVisualModule, self).__init__()
        self.dataset_type = dataset_type
        self.features_dim = features_dim
        self.hidden_size = hidden_size
        self.process = True

        if dataset_type == "features":
            if features_dim == hidden_size:
                self.process = False
            else:
                self.process_input = nn.Linear(
                    *(features_dim, hidden_size)
                    if sender
                    else (hidden_size, features_dim)
                )

        if dataset_type == "meta":
            self.process_input = nn.Linear(
                *(meta_dim, hidden_size) if sender else (hidden_size, meta_dim)
            )

    def reset_parameters(self):
        if self.process:
            self.process_input.reset_parameters()

    def forward(self, input):
        batch_size = input.shape[0]
        # reduce features to hidden, or hidden to features
        # or reduce metadata to hidden, or hidden to metadata
        if self.process and (
            self.dataset_type == "features" or self.dataset_type == "meta"
        ):
            input = self.process_input(input)

        return input


class ShapesSender(nn.Module):
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
        dataset_type="meta",
        reset_params=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.output_len = output_len
        self.sos_id = sos_id

        if eos_id is None:
            self.eos_id = sos_id
        else:
            self.eos_id = eos_id

        self.input_module = ShapesMetaVisualModule(
            hidden_size=hidden_size, dataset_type=dataset_type
        )

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.greedy = greedy

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        elif cell_type == "darts":
            self.rnn = DARTSCell(embedding_size, hidden_size, genotype)
        else:
            raise ValueError(
                "ShapesSender case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )
        self.linear_out = nn.Linear(
            hidden_size, vocab_size
        )  # from a hidden state to the vocab
        if reset_params:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

        self.input_module.reset_parameters()

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

    def _init_state(self, hidden_state, rnn_type, device):
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

    def _calculate_seq_len(self, seq_lengths, token, initial_length, seq_pos):
        """
            Calculates the lengths of each sequence in the batch in-place.
            The length goes from the start of the sequece up until the eos_id is predicted.
            If it is not predicted, then the length is output_len + n_sos_symbols.
            Args:
                seq_lengths (torch.tensor): To keep track of the sequence lengths.
                token (torch.tensor): Batch of predicted tokens at this timestep.
                initial_length (int): The max possible sequence length (output_len + n_sos_symbols).
                seq_pos (int): The current timestep.
        """
        if self.training:
            max_predicted, vocab_index = torch.max(token, dim=1)
            mask = (vocab_index == self.eos_id) * (max_predicted == 1.0)
        else:
            mask = token == self.eos_id
        mask *= seq_lengths == initial_length
        seq_lengths[mask.nonzero()] = seq_pos + 1  # start always token appended

    def forward(self, tau=1.2, hidden_state=None, device=None):
        """
        Performs a forward pass. If training, use Gumbel Softmax (hard) for sampling, else use
        discrete sampling.
        Hidden state here represents the encoded image/metadata - initializes the RNN from it.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_state = self.input_module(hidden_state)
        state, batch_size = self._init_state(hidden_state, type(self.rnn), device)

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
        initial_length = self.output_len + 1  # add the sos token
        seq_lengths = (
            torch.ones([batch_size], dtype=torch.int64, device=device) * initial_length
        )

        embeds = []  # keep track of the embedded sequence
        entropy = 0.0
        sentence_probability = torch.zeros((batch_size, self.vocab_size), device=device)

        for i in range(self.output_len):
            if self.training:
                emb = torch.matmul(output[-1], self.embedding)
            else:
                emb = self.embedding[output[-1]]

            embeds.append(emb)
            state = self.rnn(emb, state)

            if type(self.rnn) is nn.LSTMCell:
                h, c = state
            else:
                h = state

            p = F.softmax(self.linear_out(h), dim=1)
            entropy += Categorical(p).entropy()

            if self.training:
                token = gumbel_softmax(p, tau, hard=True)
            else:
                sentence_probability += p.detach()
                if self.greedy:
                    _, token = torch.max(p, -1)

                else:
                    token = Categorical(p).sample()

                if batch_size == 1:
                    token = token.unsqueeze(0)

            output.append(token)

            self._calculate_seq_len(seq_lengths, token, initial_length, seq_pos=i + 1)

        return (
            torch.stack(output, dim=1),
            seq_lengths,
            torch.mean(entropy) / self.output_len,
            torch.stack(embeds, dim=1),
            sentence_probability,
        )


class ShapesReceiver(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=256,
        hidden_size=512,
        cell_type="lstm",
        genotype=None,
        dataset_type="meta",
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        self.input_module = ShapesMetaVisualModule(
            hidden_size=hidden_size, dataset_type=dataset_type, sender=False
        )

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        elif cell_type == "darts":
            self.rnn = DARTSCell(embedding_size, hidden_size, genotype)
        else:
            raise ValueError(
                "ShapesReceiver case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)
        self.input_module.reset_parameters()
        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def forward(self, messages=None, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = messages.shape[0]

        emb = (
            torch.matmul(messages, self.embedding)
            if self.training
            else self.embedding[messages]
        )

        # initialize hidden
        h = torch.zeros([batch_size, self.hidden_size], device=device)
        if self.cell_type == "lstm":
            c = torch.zeros([batch_size, self.hidden_size], device=device)
            h = (h, c)

        # make sequence_length be first dim
        seq_iterator = emb.transpose(0, 1)
        for w in seq_iterator:
            h = self.rnn(w, h)

        if self.cell_type == "lstm":
            h = h[0]  # keep only hidden state

        out = self.input_module(h)

        return out, emb


class ShapesSingleModel(ShapesSender):
    def __init__(self, *args, **kwargs):
        kwargs["reset_params"] = False
        super().__init__(*args, **kwargs)

        self.output_module = ShapesMetaVisualModule(
            hidden_size=kwargs["hidden_size"],
            dataset_type=kwargs["dataset_type"],
            sender=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

        self.input_module.reset_parameters()
        self.output_module.reset_parameters()

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

    def forward(self, hidden_state=None, messages=None, device=None, tau=1.2):
        """
        Merged version of Sender and Receiver
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if messages is None:
            hidden_state = self.input_module(hidden_state)
            state, batch_size = self._init_state(hidden_state, type(self.rnn), device)

            # Init output
            if self.training:
                output = [
                    torch.zeros(
                        (batch_size, self.vocab_size),
                        dtype=torch.float32,
                        device=device,
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
            initial_length = self.output_len + 1  # add the sos token
            seq_lengths = (
                torch.ones([batch_size], dtype=torch.int64, device=device)
                * initial_length
            )

            embeds = []  # keep track of the embedded sequence
            entropy = 0.0
            sentence_probability = torch.zeros(
                (batch_size, self.vocab_size), device=device
            )

            for i in range(self.output_len):
                if self.training:
                    emb = torch.matmul(output[-1], self.embedding)
                else:
                    emb = self.embedding[output[-1]]

                embeds.append(emb)
                state = self.rnn(emb, state)

                if type(self.rnn) is nn.LSTMCell:
                    h, c = state
                else:
                    h = state

                p = F.softmax(self.linear_out(h), dim=1)
                entropy += Categorical(p).entropy()

                if self.training:
                    token = gumbel_softmax(p, tau, hard=True)
                else:
                    sentence_probability += p.detach()
                    if self.greedy:
                        _, token = torch.max(p, -1)

                    else:
                        token = Categorical(p).sample()

                    if batch_size == 1:
                        token = token.unsqueeze(0)

                output.append(token)

                self._calculate_seq_len(
                    seq_lengths, token, initial_length, seq_pos=i + 1
                )

            return (
                torch.stack(output, dim=1),
                seq_lengths,
                torch.mean(entropy) / self.output_len,
                torch.stack(embeds, dim=1),
                sentence_probability,
            )

        else:
            batch_size = messages.shape[0]

            emb = (
                torch.matmul(messages, self.embedding)
                if self.training
                else self.embedding[messages]
            )

            # initialize hidden
            h = torch.zeros([batch_size, self.hidden_size], device=device)
            if self.cell_type == "lstm":
                c = torch.zeros([batch_size, self.hidden_size], device=device)
                h = (h, c)

            # make sequence_length be first dim
            seq_iterator = emb.transpose(0, 1)
            for w in seq_iterator:
                h = self.rnn(w, h)

            if self.cell_type == "lstm":
                h = h[0]  # keep only hidden state

            out = self.output_module(h)

            return out, emb
