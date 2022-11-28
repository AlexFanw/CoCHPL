import math
from tqdm import tqdm
from collections import namedtuple
import argparse
from itertools import count, chain
import torch.nn as nn
import torch.optim as optim

from RL.network_advantage import AdvantageNetwork
from RL.network_dueling_Q import DuelingQNetwork
from RL.RL_memory import ReplayMemoryPER
from RL.network_termination import TerminationNetwork
from RL.recommend_env.env_variable_question import VariableRecommendEnv
from utils.utils import *
from RL.recommend_env.env_binary_question import BinaryRecommendEnv
from RL.recommend_env.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from graph.gcn import GraphEncoder
import warnings

warnings.filterwarnings("ignore")

RecommendEnv = {
    LAST_FM: VariableRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv
}
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand_items', 'next_cand_features'))


class RecAgent(object):
    def __init__(self, device, memory, action_size, hidden_size, gcn_net, learning_rate, l2_norm,
                 PADDING_ID, value_net, EPS_START=0.9, EPS_END=0.1, EPS_DECAY=0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.device = device
        # GCN+Transformer Embedding
        self.gcn_net = gcn_net
        # self.ask_agent = ask_agent
        # # Termination Network
        self.termination_net = TerminationNetwork(action_size, hidden_size)
        # # Value Network
        self.value_net = value_net
        # Action Select
        self.policy_net = AdvantageNetwork(action_size, hidden_size).to(device)
        self.target_net = AdvantageNetwork(action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(),
                                          self.gcn_net.parameters(),
                                          self.termination_net.parameters(),
                                          self.value_net.parameters()),
                                    lr=learning_rate,
                                    weight_decay=l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau

    def select_action(self, state, cand_items, items_space, is_test=False):
        state_emb = self.gcn_net([state])
        cand_features = torch.LongTensor([cand_items]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand_features)
        sample = random.random()
        eps_threshold = self.EPS_END
        '''
        Greedy Soft Policy
        '''
        if is_test or sample > eps_threshold:
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                chosen_feature = cand_features[0][actions_value.argmax().item()]
                return chosen_feature
        else:
            shuffled_cand = items_space
            random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long)

    def update_target_model(self):
        # soft assign
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, BATCH_SIZE, GAMMA, ask_agent=None):
        if len(self.memory) < BATCH_SIZE:
            return

        self.update_target_model()

        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        n_states = []
        n_cand_features = []
        n_cand_items = []
        for s, ci, cf in zip(batch.next_state, batch.next_cand_items, batch.next_cand_features):
            if s is not None:
                n_states.append(s)
                n_cand_items.append(ci)
                n_cand_features.append(cf)
        if not n_states:
            return 0

        '''
        Double DQN
        Q_policy - (r + GAMMA * [max Q_target(next) (1-termination) + Value (termination)])
        '''
        _, q_target_features = self.calculate_q_score(BATCH_SIZE, batch, n_states, n_cand_features, ask_agent)

        q_policy_items, q_target_items = self.calculate_q_score(BATCH_SIZE, batch, n_states, n_cand_items)

        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cand_items)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)
        termination = self.termination_net(next_state_emb_batch, next_cand_emb_batch)

        reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)

        q_max = torch.maximum(q_target_features, q_target_items)

        q_target = reward_batch + GAMMA * ((1-termination) * q_target_items + termination * q_max)

        # prioritized experience replay
        errors = (q_policy_items - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        # mean squared error loss to minimize
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_policy_items, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data

    def calculate_q_score(self, BATCH_SIZE, batch, n_states, n_cands, ask_agent=None):
        if ask_agent == None:
            state_emb_batch = self.gcn_net(list(batch.state))
            action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device)  # [N*1]

            action_emb_batch = self.gcn_net.embedding(action_batch)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.uint8)

            next_state_emb_batch = self.gcn_net(n_states)
            next_cand_batch = self.padding(n_cands)
            next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)

            value = self.value_net(state_emb_batch)
            q_policy = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False) + value

            best_next_actions = torch.gather(input=next_cand_batch, dim=1,
                                             index=self.policy_net(next_state_emb_batch, next_cand_emb_batch).argmax(
                                                 dim=1).view(
                                                 len(n_states), 1).to(self.device))
            best_next_actions_emb = self.gcn_net.embedding(best_next_actions)
            q_target = torch.zeros((BATCH_SIZE, 1), device=self.device)
            q_target[non_final_mask] = self.target_net(next_state_emb_batch, best_next_actions_emb,
                                                       choose_action=False).detach()
            q_target += value
            return q_policy, q_target
        else:
            with torch.no_grad():
                state_emb_batch = ask_agent.gcn_net(list(batch.state))
                action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(ask_agent.device)  # [N*1]

                action_emb_batch = ask_agent.gcn_net.embedding(action_batch)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=ask_agent.device, dtype=torch.uint8)

                next_state_emb_batch = ask_agent.gcn_net(n_states)
                next_cand_batch = ask_agent.padding(n_cands)
                next_cand_emb_batch = ask_agent.gcn_net.embedding(next_cand_batch)

                value = ask_agent.value_net(state_emb_batch)
                q_policy = ask_agent.policy_net(state_emb_batch, action_emb_batch, choose_action=False) + value

                best_next_actions = torch.gather(input=next_cand_batch, dim=1,
                                                 index=ask_agent.policy_net(next_state_emb_batch, next_cand_emb_batch).argmax(
                                                     dim=1).view(
                                                     len(n_states), 1).to(ask_agent.device))
                best_next_actions_emb = ask_agent.gcn_net.embedding(best_next_actions)
                q_target = torch.zeros((BATCH_SIZE, 1), device=ask_agent.device)
                q_target[non_final_mask] = ask_agent.target_net(next_state_emb_batch, best_next_actions_emb,
                                                           choose_action=False).detach()
                q_target += value
                return q_policy, q_target

    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name,
                      model={'policy': self.policy_net.state_dict(),
                             'gcn': self.gcn_net.state_dict(),
                             'termination': self.termination_net.state_dict()},
                      filename=filename, epoch_user=epoch_user, agent='rec')

    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user, agent='rec')
        self.policy_net.load_state_dict(model_dict['policy'])
        self.gcn_net.load_state_dict(model_dict['gcn'])
        self.termination_net.load_state_dict(model_dict['termination'])

    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)