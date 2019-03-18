import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):
    def __init__(self, sender, receiver, tau=1.2):
        super().__init__()

        self.sender = sender
        self.receiver = receiver
        self.tau = tau

    def forward(self, target, distractors):
        batch_size = len(target)

        target = target.to(device)
        distractors = [d.to(device) for d in distractors]

        messages, lengths = self.sender(self.tau, hidden_state=target)
        r_transform = self.receiver(messages, lengths)

        loss = 0

        target = target.view(batch_size, 1, -1)
        r_transform = r_transform.view(batch_size, -1, 1)

        target_score = torch.bmm(target, r_transform).squeeze()  # scalar

        distractors_scores = []

        for d in distractors:
            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            distractors_scores.append(d_score)
            zero_tensor = torch.tensor(0.0, device=device)

            loss += torch.max(zero_tensor, 1.0 - target_score + d_score)

        # Calculate accuracy
        all_scores = torch.zeros((batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        for i, score in enumerate(distractors_scores):
            all_scores[:, i + 1] = score

        all_scores = torch.exp(all_scores)

        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        return torch.mean(loss), torch.mean(accuracy), messages
