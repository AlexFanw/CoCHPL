import math
from tqdm import tqdm
from collections import namedtuple
import argparse
from itertools import count, chain
import torch.nn as nn
import torch.optim as optim

from RL.agent.ask_agent import AskAgent
from RL.agent.rec_agent import RecAgent
from RL.network_dueling_Q import DuelingQNetwork
from RL.RL_memory import ReplayMemoryPER
from RL.network_value import ValueNetwork
from utils.utils import *
from RL.recommend_env.env_binary_question import BinaryRecommendEnv
from RL.recommend_env.env_enumerated_question import EnumeratedRecommendEnv
from RL.recommend_env.env_variable_question import VariableRecommendEnv
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
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))


def choose_option(ask_agent, rec_agent, state, cand):
    state_emb = ask_agent.gcn_net([state])
    feature_cand = cand["feature"]
    ask_score = []
    for feature in feature_cand:
        feature = torch.LongTensor(np.array(feature).astype(int).reshape(-1, 1)).to(ask_agent.device)  # [N*1]
        feature = ask_agent.gcn_net.embedding(feature)
        ask_score.append(ask_agent.policy_net(state_emb, feature, choose_action=False).detach().numpy().squeeze())
    ask_Q = np.array(ask_score).dot(np.exp(ask_score) / sum(np.exp(ask_score)))
    # print()

    state_emb = rec_agent.gcn_net([state])
    item_cand = cand["item"]
    rec_score = []
    for item in item_cand:
        item = torch.LongTensor(np.array(item).astype(int).reshape(-1, 1)).to(rec_agent.device)  # [N*1]
        item = rec_agent.gcn_net.embedding(item)
        rec_score.append(rec_agent.policy_net(state_emb, item, choose_action=False).detach().numpy().squeeze())
    rec_Q = np.array(rec_score).dot(np.exp(rec_score) / sum(np.exp(rec_score)))
    if ask_Q > rec_Q:
        return 1
    else:
        return 0


def option_critic_train(args, kg, dataset, filename):
    """RL Model Train

    :param args: some experiment settings
    :param kg: knowledge graph
    :param dataset: dataset
    :param filename: training model file saving path
    :return:
    """
    set_random_seed(args.seed)
    # Prepare the Environment
    env = RecommendEnv[args.data_name](kg, dataset,
                                       args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num,
                                       entropy_way=args.entropy_method, max_step=args.max_step)

    # User&Feature Embedding
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1, env.ui_embeds.shape[1]))), axis=0))

    '''
    VALUE NET
    '''
    value_net = ValueNetwork(action_size=embed.size(1)).to(args.device)
    '''
    ASK AGENT
    '''
    # ASK GCN Transformer
    ask_gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
                               embeddings=embed,
                               fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)
    # Ask Memory
    ask_memory = ReplayMemoryPER(args.memory_size)  # 50000
    # Ask Agent
    ask_agent = AskAgent(device=args.device, memory=ask_memory, action_size=embed.size(1),
                         hidden_size=args.hidden, gcn_net=ask_gcn_net, learning_rate=args.learning_rate,
                         l2_norm=args.l2_norm, PADDING_ID=embed.size(0) - 1, value_net=value_net)
    '''
    REC AGENT
    '''
    # Rec GCN Transformer
    rec_gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
                               embeddings=embed,
                               fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)
    # Rec Memory
    rec_memory = ReplayMemoryPER(args.memory_size)  # 50000
    # Rec Agent
    rec_agent = RecAgent(device=args.device, memory=rec_memory, action_size=embed.size(1),
                         hidden_size=args.hidden, gcn_net=rec_gcn_net, learning_rate=args.learning_rate,
                         l2_norm=args.l2_norm, PADDING_ID=embed.size(0) - 1, value_net=value_net)
    # load parameters
    if args.load_rl_epoch != 0:
        print('Loading Model in epoch {}'.format(args.load_rl_epoch))
        ask_agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
        rec_agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
        value_net.load_value_net(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
        # TODO: gru_net.load_model()

    test_performance = []
    # TODO Evaluate
    # if args.eval_num == 1:
    #     SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
    #     test_performance.append(SR15_mean)
    for epoch in range(1 + args.load_rl_epoch, args.max_epoch + 1):
        print(">>>Train Step: {}, Total: {}".format(epoch, args.max_epoch))
        SR5, SR10, SR15, AvgT, HDCG, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times), desc='sampling'):
            # blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            # if not args.fix_emb:
            #     # Reset environment and record the starting state
            #     # state, cand, action_space = env.reset(
            #     #     agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
            #     # TODO env reset
            # else:
            state, cand, action_space = env.reset()
            # state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
            epi_reward = 0
            is_last_turn = False
            done = 0
            for t in count():  # Turn
                if t == 14:
                    is_last_turn = True
                '''
                Select Option
                '''
                if done:
                    break
                option = choose_option(ask_agent, rec_agent, state, cand)
                if option:
                    print("\n————————Turn: ", t, "  Option: ASK————————")
                else:
                    print("\n————————Turn: ", t, "  Option: REC————————")
                # ASK
                # env.cur_conver_step = 0
                if option == 1:
                    term = False

                    while not term and not done:
                        '''
                        Select Action
                        '''
                        # print(state, cand["feature"], action_space["feature"])
                        action, sorted_actions = ask_agent.select_action(state, cand["feature"], action_space["feature"], is_last_turn=is_last_turn)
                        '''
                        Env Interaction
                        '''
                        # if not args.fix_emb:
                        #     next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions,
                        #                                                          agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                        # else:
                        next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions)
                        epi_reward += reward
                        reward = torch.tensor([reward], device=args.device, dtype=torch.float)

                        # Whether Termination
                        next_state_emb = ask_agent.gcn_net([next_state])
                        next_cand_emb = ask_agent.gcn_net.embedding(torch.LongTensor([cand["feature"]]).to(args.device))
                        term_score = ask_agent.termination_net(next_state_emb, next_cand_emb)
                        if term_score >= 0.5:
                            term = True

                        if done:
                            next_state = None

                        ask_agent.memory.push(state, action, next_state, reward, next_cand["feature"])
                        state = next_state
                        cand = next_cand

                        newloss = ask_agent.optimize_model(args.batch_size, args.gamma)
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
                                HDCG += (1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(
                                    done + 1, 2))
                            else:
                                HDCG += 0
                            AvgT += t + 1
                            total_reward += epi_reward
                            break

                    env.cur_conver_turn += 1
                # RECOMMEND
                elif option == 0:
                    term = False
                    while not term and not done:
                        '''
                        Select Action
                        '''
                        # state_emb = ask_agent.gcn_net([state])
                        # cand_emb = ask_agent.gcn_net.embedding(torch.LongTensor([cand["feature"]]).to(args.device))
                        action, sortxed_actions = rec_agent.select_action(state, cand["item"], action_space["item"], is_last_turn=is_last_turn)
                        '''
                        Env Interaction
                        '''
                        # if not args.fix_emb:
                        #     next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions,
                        #                                                          agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                        # else:
                        next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions)
                        epi_reward += reward
                        reward = torch.tensor([reward], device=args.device, dtype=torch.float)

                        # Whether Termination
                        next_state_emb = rec_agent.gcn_net([next_state])
                        next_cand_emb = rec_agent.gcn_net.embedding(
                            torch.LongTensor([cand["item"]]).to(args.device))
                        term_score = rec_agent.termination_net(next_state_emb, next_cand_emb)
                        if term_score >= 0.5:
                            term = True

                        if done:
                            next_state = None

                        rec_agent.memory.push(state, action, next_state, reward, next_cand["item"])
                        state = next_state
                        cand = next_cand

                        newloss = rec_agent.optimize_model(args.batch_size, args.gamma)
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
                                HDCG += (1 / math.log(t + 3, 2) + (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) / math.log(
                                    done + 1, 2))
                            else:
                                HDCG += 0
                            AvgT += t + 1
                            total_reward += epi_reward
                            break

                    env.cur_conver_turn += 1
        enablePrint()  # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item() / args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, HDCG:{}, rewards:{} '
              'Total epoch_uesr:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                           AvgT / args.sample_times, HDCG / args.sample_times,
                                           total_reward / args.sample_times, args.sample_times))

        # if epoch % args.eval_num == 0:
        #     SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, epoch)
        #     test_performance.append(SR15_mean)
        if epoch % args.save_num == 0:
            ask_agent.save_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
            rec_agent.save_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
            value_net.save_value_net(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
            # TODO: gru_net.save_model()
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
    parser.add_argument('--max_step', type=int, default=50, help='max conversation turn')
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