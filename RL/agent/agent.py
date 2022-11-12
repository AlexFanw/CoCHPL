import math
from tqdm import tqdm
from collections import namedtuple
import argparse
from itertools import count, chain
import torch.nn as nn
import torch.optim as optim

from RL.network_dueling_Q import DuelingQNetwork
from RL.RL_memory import ReplayMemoryPER
from utils.utils import *
from RL.recommend_env.env_binary_question import BinaryRecommendEnv
from RL.recommend_env.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from graph.gcn import GraphEncoder
import warnings
warnings.filterwarnings("ignore")

RecommendEnv = {
    LAST_FM: BinaryRecommendEnv,
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
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))

class Agent(object):
    def __init__(self, device, memory, action_size, hidden_size, gcn_net, learning_rate, l2_norm,
                 PADDING_ID, EPS_START=0.9, EPS_END=0.1, EPS_DECAY=0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DuelingQNetwork(action_size, hidden_size).to(device)
        self.target_net = DuelingQNetwork(action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(), self.gcn_net.parameters()), lr=learning_rate,
                                    weight_decay=l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau

    def select_action(self, state, cand, action_space, is_test=False, is_last_turn=False):
        state_emb = self.gcn_net([state])
        cand = torch.LongTensor([cand]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        '''
        Greedy soft policy
        '''
        if is_test or sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1]
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                print(sorted(list(zip(cand[0].tolist(), actions_value[0].tolist())), key=lambda x: x[1], reverse=True))
                action = cand[0][actions_value.argmax().item()]
                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist()
        else:
            shuffled_cand = action_space[0] + action_space[1]
            random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand

    def update_target_model(self):
        # soft assign
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return

        self.update_target_model()

        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device)  # [N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        for s, c in zip(batch.next_state, batch.next_cand):
            if s is not None:
                n_states.append(s)
                n_cands.append(c)
        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)

        '''
        Double DQN
        Q_policy - GAMMA * (r + max Q_target(next))
        '''
        q_policy = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)
        
        best_next_actions = torch.gather(input=next_cand_batch, dim=1,
                                    index=self.policy_net(next_state_emb_batch, next_cand_emb_batch).argmax(dim=1).view(
                                        len(n_states), 1).to(self.device))
        best_next_actions_emb = self.gcn_net.embedding(best_next_actions)
        q_target = torch.zeros((BATCH_SIZE, 1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch, best_next_actions_emb, choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target

        # prioritized experience replay
        errors = (q_policy - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        # mean squared error loss to minimize
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_policy, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data

    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name,
                      model={'policy': self.policy_net.state_dict(), 'gcn': self.gcn_net.state_dict()},
                      filename=filename, epoch_user=epoch_user)

    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict['policy'])
        self.gcn_net.load_state_dict(model_dict['gcn'])

    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)

def train(args, kg, dataset, filename):
    """RL Model Train

    :param args: some experiment settings
    :param kg: knowledge graph
    :param dataset: dataset
    :param filename: training model file saving path
    :return:
    """
    set_random_seed(args.seed)
    env = RecommendEnv[args.data_name](kg, dataset,
                                       args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num,
                                       entropy_way=args.entropy_method)

    # User&Feature Embedding
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1, env.ui_embeds.shape[1]))), axis=0))
    # GCN Transformer
    gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed,
                           fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)
    # Memory
    memory = ReplayMemoryPER(args.memory_size)  # 50000
    # Recommendation Agent
    agent = Agent(device=args.device, memory=memory, action_size=embed.size(1),
                  hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm,
                  PADDING_ID=embed.size(0) - 1)

    # evaluation metric  ST@T
    # agent load policy parameters
    if args.load_rl_epoch != 0:
        print('Loading RL Agent in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 1:
        SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
        test_performance.append(SR15_mean)
    for train_step in range(1 + args.load_rl_epoch, args.max_epoch + 1):
        print(">>>Train Step: {}, Total: {}".format(train_step, args.max_epoch))
        SR5, SR10, SR15, AvgT, Rank, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times), desc='sampling'):
            # blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            if not args.fix_emb:
                # Reset environment and record the starting state
                state, cand, action_space = env.reset(
                    agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
            else:
                state, cand, action_space = env.reset()
                # state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
            epi_reward = 0
            is_last_turn = False
            for t in count():  # user  dialog
                if t == 14:
                    is_last_turn = True
                '''
                Select Action
                '''
                action, sorted_actions = agent.select_action(state, cand, action_space, is_last_turn=is_last_turn)
                '''
                Env Interaction
                '''
                if not args.fix_emb:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions,
                                                                                 agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                else:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                if done:
                    next_state = None

                agent.memory.push(state, action, next_state, reward, next_cand)
                state = next_state
                cand = next_cand

                newloss = agent.optimize_model(args.batch_size, args.gamma)
                if newloss is not None:
                    loss += newloss

                if done:
                    # every episode update the target model to be same with model
                    if reward.item() == 1:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += (1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(
                            done + 1, 2))
                        print((1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(
                            done + 1, 2)))
                    else:
                        Rank += 0
                    AvgT += t + 1
                    total_reward += epi_reward
                    break
        enablePrint()  # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item() / args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, rewards:{} '
              'Total epoch_uesr:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                           AvgT / args.sample_times, Rank / args.sample_times,
                                           total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, train_step)
            test_performance.append(SR15_mean)
        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)

def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    # parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    # parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')
    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='weight_entropy',
                        help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the episodes of sampling')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--eval_num', type=int, default=10, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=10, help='the number of steps to save RL model and metric')
    parser.add_argument('--observe_num', type=int, default=1000, help='the number of steps to print metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'],
                        help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    args = parser.parse_args()
    return args