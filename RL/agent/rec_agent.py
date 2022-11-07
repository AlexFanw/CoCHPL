from RL.agent.agent import Agent


class RecAgent(object):
    def __init__(self, device, memory, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID):
        super().__init__(device, memory, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID)