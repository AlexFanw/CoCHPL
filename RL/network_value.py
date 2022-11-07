import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")


class ValueNetwork(nn.Module):
    def __init__(self, hidden_size=100):
        super(ValueNetwork, self).__init__()
        # V(s)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        :param x: encode history [N*L*D]
        :return: v: action score [N*K]
        """
        # V(s)
        x = F.relu(self.linear(x))
        value = self.value(x).squeeze(dim=2)  # [N*1*1]
        return value