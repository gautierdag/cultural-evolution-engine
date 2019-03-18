import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):
    def __init__(self, sender, receiver):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

    def forward(self, target, distractors, tau=1.2):
        batch_size = target.shape[0]

        target = target.to(device)
        distractors = [d.to(device) for d in distractors]

        messages = self.sender(tau, hidden_state=target)
        r_transform = self.receiver(messages)

        loss = 0

        target = target.view(batch_size, 1, -1)
        r_transform = r_transform.view(batch_size, -1, 1)

        target_score = torch.bmm(target, r_transform).squeeze()  # scalar

        all_scores = torch.zeros((batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        for i, d in enumerate(distractors):
            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            all_scores[:, i + 1] = d_score
            loss += torch.max(
                torch.tensor(0.0, device=device), 1.0 - target_score + d_score
            )

        # Calculate accuracy
        all_scores = torch.exp(all_scores)
        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        return torch.mean(loss), torch.mean(accuracy), messages
