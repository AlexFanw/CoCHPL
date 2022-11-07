import torch
import torch.nn as nn


class IntraOptionNetwork(nn.Module):
    """
    1.Decide whether to ask or rec.

    2.Terminal Function. If x>=0.5, ask. x < 0.5, rec
    """
    def __init__(self, state_emb_size, hidden_size):
        super(IntraOptionNetwork, self).__init__()
        self.hidden = torch.nn.Linear(state_emb_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

