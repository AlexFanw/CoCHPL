import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")


class DuelingQNetwork(nn.Module):
    def __init__(self, action_size, hidden_size=100):
        super(DuelingQNetwork, self).__init__()
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)
        self.out_advantage = nn.Linear(hidden_size, 1)

    def forward(self, x, y, choose_action=True):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        # V(s)
        fc2_value = F.relu(self.fc2_value(x))
        value = self.out_value(fc2_value).squeeze(dim=2)  # [N*1*1]
        # Q(s,a)
        x = x.repeat(1, y.size(1), 1)
        state_cat_action = torch.cat((x, y), dim=2)
        advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2)  # [N*K]

        if choose_action:
            mean_adv = advantage.mean(dim=1, keepdim=True)
            qsa = advantage + value - mean_adv
        else:
            qsa = advantage + value
        return qsa