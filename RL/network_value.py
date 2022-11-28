import torch
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
        :return: v: Value Score [N*1]
        """
        # V(s)
        x = F.relu(self.linear(x))
        value = self.value(x).squeeze(dim=2)  # [N*1*1]
        return value

    def load_value_net(self, data_name, filename, epoch_user):
        model_file = CHECKPOINT_DIR[data_name] + '/model/' + 'value-' + filename + '-epoch-{}.pkl'.format(epoch_user)
        model_dict = torch.load(model_file)
        print('Value model load at {}'.format(model_file))
        self.load_state_dict(model_dict["value"])

    def save_value_net(self, data_name, filename, epoch_user):
        model = {'value': self.state_dict()}
        model_file = CHECKPOINT_DIR[data_name] + '/model/' + 'value-' + filename + '-epoch-{}.pkl'.format(epoch_user)
        if not os.path.isdir(CHECKPOINT_DIR[data_name] + '/model/'):
            os.makedirs(CHECKPOINT_DIR[data_name] + '/model/')
        torch.save(model, model_file)
        print('Value model saved at {}'.format(model_file))


if __name__ == "__main__":
    vn = ValueNetwork(1, 2)
    print(vn(torch.Tensor([[[1, 2], [1, 2]]])))
