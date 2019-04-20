import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from .gumbel import gumbel_softmax
from .DARTSCell import DARTSCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObverterMetaVisualModule(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        dataset_type="meta",
        in_features=1000,
        object_vocab_size=None,
        color_vocab_size=None,
    ):
        super(ObverterMetaVisualModule, self).__init__()
        self.dataset_type = dataset_type
        self.hidden_size = hidden_size

        if dataset_type == "raw":
            n_filters = 20
            self.process_input = nn.Sequential(
                nn.Conv2d(3, n_filters, 3, stride=2),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, 3, stride=2),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, 3, stride=2),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, 3, stride=2),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, 3, stride=2),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
            )
            self.linear_out = nn.Linear(180, hidden_size)

        if dataset_type == "features":
            self.process_input = nn.Linear(in_features, hidden_size)

        if dataset_type == "meta":
            self.process_input_color = nn.Embedding(
                color_vocab_size, int(hidden_size / 2)
            )
            self.process_input_object = nn.Embedding(
                object_vocab_size, int(hidden_size / 2)
            )

        if dataset_type == "meta_combined":
            self.process_input == nn.Embedding(object_vocab_size, int(hidden_size / 2))

    def forward(self, input):
        batch_size = input.shape[0]
        # process raw image using conv net
        if self.dataset_type == "raw":
            img_enc = self.process_input(input.transpose(1, 3)).view(batch_size, -1)
            embedding = self.linear_out(img_enc)
        # process precomputed features by reducing them to hidden size
        if self.dataset_type == "features":
            embedding = self.process_input(input)
        # process metadata input with separate vocab for color/object
        if self.dataset_type == "meta":
            object_embedding = self.process_input_object(input[:, 0])
            color_embedding = self.process_input_color(input[:, 1])
            embedding = torch.cat((object_embedding, color_embedding), 1)

        # process metadata input with same vocab for color/object
        if self.dataset_type == "meta_combined":
            embedding = self.process_input(input).view(batch_size, -1)

        assert embedding.shape[1] == self.hidden_size
        return embedding


class ObverterSender(nn.Module):
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
        in_features=1000,
        object_vocab_size=None,
        color_vocab_size=None,
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

        self.obverter_module = ObverterMetaVisualModule(
            hidden_size=hidden_size,
            dataset_type=dataset_type,
            in_features=in_features,
            object_vocab_size=object_vocab_size,
            color_vocab_size=color_vocab_size,
        )

        # rnn settings
        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        elif cell_type == "darts":
            self.rnn = DARTSCell(embedding_size, hidden_size, genotype)
        else:
            raise ValueError(
                "ObverterSender case with cell_type '{}' is undefined".format(cell_type)
            )

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

    def forward(self, image_representation, tau):
        """
        Performs a forward pass. If training, use Gumbel Softmax (hard) for sampling, else use
        discrete sampling.
        Hidden state here represents the encoded image/metadata/features - initializes the RNN from it.
        """

        image_representation = self.obverter_module(image_representation)

        # initialize the rnn using the obverter module
        state, batch_size = self._init_state(image_representation, type(self.rnn))

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
        )
