import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")


class AdvantageNetwork(nn.Module):
    def __init__(self, action_size, hidden_size=100):
        super(AdvantageNetwork, self).__init__()
        # Q(s,a)
        self.advantage = nn.Linear(hidden_size + action_size, hidden_size)
        self.out_advantage = nn.Linear(hidden_size, 1)

    def forward(self, x, y, choose_action=True):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        # Q(s,a)
        x = x.repeat(1, y.size(1), 1)
        state_cat_action = torch.cat((x, y), dim=2)
        advantage = self.out_advantage(F.relu(self.advantage(state_cat_action))).squeeze(dim=2)  # [N*K]

        if choose_action:
            mean_adv = advantage.mean(dim=1, keepdim=True)
            qsa = advantage - mean_adv
        else:
            qsa = advantage
        return qsa