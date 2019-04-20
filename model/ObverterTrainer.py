import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObverterTrainer(nn.Module):
    def __init__(self, sender, receiver):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        batch_size, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=device).expand(
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

    def forward(self, target, distractors, tau=1.2):
        pass

    def decode(model, all_inputs, max_sentence_len, vocab_size, device):
        relevant_procs = list(range(all_inputs.size(0)))

        actions = np.array(
            [[-1 for _ in range(max_sentence_len)] for _ in relevant_procs]
        )
        all_probs = np.array([-1.0 for _ in relevant_procs])

        for l in range(max_sentence_len):
            inputs = all_inputs[relevant_procs]
            batch_size = inputs.size(0)
            next_symbol = np.tile(
                np.expand_dims(np.arange(0, vocab_size), 1), batch_size
            ).transpose()

            if l > 0:
                run_communications = np.concatenate(
                    (
                        np.expand_dims(
                            actions[relevant_procs, :l].transpose(), 2
                        ).repeat(vocab_size, axis=2),
                        np.expand_dims(next_symbol, 0),
                    ),
                    axis=0,
                )
            else:
                run_communications = np.expand_dims(next_symbol, 0)

            expanded_inputs = inputs.repeat(vocab_size, 1, 1, 1)

            logits, probs = model(
                expanded_inputs,
                torch.Tensor(run_communications.transpose().reshape(-1, 1 + l))
                .long()
                .to(device),
            )
            probs = probs.view((vocab_size, batch_size)).transpose(0, 1)

            probs, sel_comm_idx = torch.max(probs, dim=1)

            comm = run_communications[
                :, np.arange(len(relevant_procs)), sel_comm_idx.data.cpu().numpy()
            ].transpose()
            finished_p = []
            for i, (action, p, prob) in enumerate(zip(comm, relevant_procs, probs)):
                if prob > 0.95:
                    finished_p.append(p)
                    if prob.item() < 0:
                        continue

                for j, symb in enumerate(action):
                    actions[p][j] = symb

                all_probs[p] = prob

            for p in finished_p:
                relevant_procs.remove(p)

            if len(relevant_procs) == 0:
                break

        actions[actions == -1] = vocab_size  # padding token
        actions = torch.Tensor(np.array(actions)).long().to(device)
        return actions, all_probs
