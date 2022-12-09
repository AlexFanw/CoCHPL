import math

from tqdm import tqdm
from collections import namedtuple
import argparse
import statistics

from RL.agent.ask_agent import AskAgent
from RL.agent.rec_agent import RecAgent
from RL.RL_memory import ReplayMemoryPER
from RL.network.network_value import ValueNetwork
from utils.utils import *
from RL.recommend_env.env_variable_question import VariableRecommendEnv
from RL.RL_evaluate import evaluate
from graph.gcn import GraphEncoder
import warnings

warnings.filterwarnings("ignore")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))


def choose_option(ask_agent, rec_agent, state, cand):
    if cand["feature"] == [] or len(cand["item"]) < 10:
        return 0, 1  # Recommend
    with torch.no_grad():
        state_emb = ask_agent.gcn_net([state])
        feature_cand = cand["feature"]
        ask_score = []
        for feature in feature_cand:
            feature = torch.LongTensor(np.array(feature).astype(int).reshape(-1, 1)).to(ask_agent.device)  # [N*1]
            feature = ask_agent.gcn_net.embedding(feature)
            # print(ask_agent.value_net(state_emb))
            ask_score.append(
                ask_agent.value_net(state_emb).detach().numpy().squeeze() + ask_agent.policy_net(state_emb, feature,
                                                                                                 choose_action=False).detach().numpy().squeeze())
        ask_Q = np.array(ask_score).dot(np.exp(ask_score) / sum(np.exp(ask_score)))

        state_emb = rec_agent.gcn_net([state])
        item_cand = cand["item"]
        rec_score = []
        for item in item_cand:
            item = torch.LongTensor(np.array(item).astype(int).reshape(-1, 1)).to(rec_agent.device)  # [N*1]
            item = rec_agent.gcn_net.embedding(item)
            rec_score.append(
                rec_agent.value_net(state_emb).detach().numpy().squeeze() + rec_agent.policy_net(state_emb, item,
                                                                                                 choose_action=False).detach().numpy().squeeze())
            rec_score.append(
                rec_agent.value_net(state_emb).detach().numpy().squeeze() + rec_agent.policy_net(state_emb, item,
                                                                                                 choose_action=False).detach().numpy().squeeze())
        rec_Q = np.array(rec_score).dot(np.exp(rec_score) / sum(np.exp(rec_score)))
        # return ask_Q / (ask_Q + rec_Q), rec_Q / (ask_Q + rec_Q)
        return math.exp(ask_Q) / (math.exp(ask_Q) + math.exp(rec_Q)), math.exp(rec_Q) / (
                math.exp(ask_Q) + math.exp(rec_Q))


def option_critic_pipeline(args, kg, dataset, filename):
    """RL Model Train

    :param args: some experiment settings
    :param kg: knowledge graph
    :param dataset: dataset
    :param filename: training model file saving path
    :return:
    """
    set_random_seed(args.seed)
    # Prepare the Environment
    env = VariableRecommendEnv(kg, dataset,
                               args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                               cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                               attr_num=args.attr_num, mode='train',
                               entropy_way=args.entropy_method, max_step=args.max_step)

    # User&Feature Embedding
    embed = torch.FloatTensor(
        np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1, env.ui_embeds.shape[1]))), axis=0))
    # print(embed.size(0), embed.size(1))
    '''
    VALUE NET
    '''
    value_net = ValueNetwork().to(args.device)
    '''
    ASK AGENT
    '''
    # ASK GCN Transformer
    ask_gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
                               embeddings=embed, fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn,
                               hidden_size=args.hidden).to(args.device)
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
                               embeddings=embed, fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn,
                               hidden_size=args.hidden).to(args.device)
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
    for epoch in range(1 + args.load_rl_epoch, args.max_epoch + 1):
        print("\nEpoch: {}, Total: {}".format(epoch, args.max_epoch))
        SR5, SR10, SR15, AvgT, HDCG_item, total_reward = 0., 0., 0., 0., 0., 0.
        rec_step_list = []
        ask_step_list = []
        HDCG_attribute_list = []
        rec_loss = []
        ask_loss = []
        rec_state_infer_loss = []
        ask_state_infer_loss = []
        for i_episode in tqdm(range(args.sample_times), desc='sampling'):
            blockPrint()
            print('\n================Epoch:{} Episode:{}===================='.format(epoch, i_episode))
            state, cand, action_space = env.reset()
            epi_reward = 0
            done = 0
            for t in range(1, 16):  # Turn
                '''
                Option choose : Select Ask / Rec
                '''
                env.cur_conver_step = 1
                if done:
                    break
                print("Candidate: ", cand)
                ask_Q, rec_Q = choose_option(ask_agent, rec_agent, state, cand)
                print(ask_Q, rec_Q)
                if ask_Q > rec_Q:
                    option = 1
                    print("\n————————Turn: ", t, "  Option: ASK————————")
                else:
                    option = 0
                    print("\n————————Turn: ", t, "  Option: REC————————")

                '''
                Intra Option choose: Select features / items
                '''
                # ASK
                if option == 1:
                    termination = False
                    ask_score = []
                    while not termination and not done:
                        if env.cur_conver_step > args.max_ask_step:
                            break

                        # Select Action
                        chosen_feature = ask_agent.select_action(state, cand["feature"], action_space["feature"])
                        # Env Interaction
                        next_state, next_cand, action_space, reward, done = env.step(chosen_feature.item(), None)
                        epi_reward += reward
                        ask_score.append(reward)
                        reward = torch.tensor([reward], device=args.device, dtype=torch.float)

                        # Whether Termination
                        next_state_emb = ask_agent.gcn_net([next_state])
                        next_cand_emb = ask_agent.gcn_net.embedding(
                            torch.LongTensor([next_cand["feature"]]).to(args.device))
                        term_score = rec_agent.termination_net(next_state_emb, next_cand_emb).item()
                        print("Termination Score:", term_score)
                        if term_score >= 0.5:
                            termination = True
                        if next_cand["feature"] == []:
                            termination = True

                        # Push memory
                        if done:
                            next_state = None

                        ask_agent.memory.push(state, chosen_feature, next_state, reward,
                                              next_cand["item"], next_cand["feature"])
                        state = next_state
                        cand = next_cand

                        if done:
                            AvgT += t
                            total_reward += epi_reward
                            break

                    # calculate HDCG Attribute
                    for i in range(len(ask_score)):
                        if ask_score[i] > 0:
                            HDCG_attribute_list.append(
                                (1 / math.log(t + 2, 2) + (1 / math.log(t + 1, 2) - 1 / math.log(t + 2, 2)) /
                                 math.log(i + 2, 2)))

                # RECOMMEND
                elif option == 0:
                    termination = False
                    items = []
                    last_step = False
                    while not termination and not done and not last_step:
                        if env.cur_conver_step == args.max_rec_step:
                            last_step = True

                        # Select Action
                        chosen_item = rec_agent.select_action(state, cand["item"], action_space["item"])
                        items.append(chosen_item.item())

                        # Env Interaction
                        next_state, next_cand, action_space, reward, done = env.step(None, items, mode="train")
                        epi_reward += reward
                        reward = torch.tensor([reward], device=args.device, dtype=torch.float)

                        # Whether Termination
                        next_state_emb = rec_agent.gcn_net([next_state])
                        next_cand_emb = rec_agent.gcn_net.embedding(
                            torch.LongTensor([next_cand["item"]]).to(args.device))
                        term_score = rec_agent.termination_net(next_state_emb, next_cand_emb).item()
                        print("Termination Score:", term_score)
                        if term_score >= 0.5:
                            termination = True

                        # Push memory
                        if done:
                            next_state = None

                        rec_agent.memory.push(state, torch.tensor(chosen_item), next_state, reward,
                                              next_cand["item"], next_cand["feature"])
                        state = next_state
                        cand = next_cand
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
                                HDCG_item += (
                                        1 / math.log(t + 2, 2) + (1 / math.log(t + 1, 2) - 1 / math.log(t + 2, 2)) /
                                        math.log(done + 1, 2))

                            AvgT += t
                            total_reward += epi_reward
                            break

                # Optimize Model
                loss, loss_state = ask_agent.optimize_model(args.batch_size, args.gamma, rec_agent)
                if loss is not None:
                    ask_loss.append(loss)
                    ask_state_infer_loss.append(loss_state)
                # ——————————————

                # Optimize Model
                loss, loss_state = rec_agent.optimize_model(args.batch_size, args.gamma, ask_agent)
                if loss is not None:
                    rec_loss.append(loss)
                    rec_state_infer_loss.append(loss_state)
                # ——————————————

                if option == 1:
                    ask_step_list.append(env.cur_conver_step - 1)
                else:
                    rec_step_list.append(env.cur_conver_step - 1)

                env.cur_conver_turn += 1

        enablePrint()  # Enable print function
        print('\nSample Times:{}'.format(args.sample_times))
        print('Recommend loss : {}'.format(statistics.mean(rec_loss)))
        print('Recommend State Infer loss : {}'.format(statistics.mean(rec_state_infer_loss)))
        print('Ask loss : {}'.format(statistics.mean(ask_loss)))
        print('Ask State Infer loss : {}'.format(statistics.mean(ask_state_infer_loss)))
        print('SR5:{}\nSR10:{}\nSR15:{}\nHDCG_item:{}\nHDCG_attribute:{}\nrewards:{}\n'.format(SR5 / args.sample_times,
                                                                                               SR10 / args.sample_times,
                                                                                               SR15 / args.sample_times,
                                                                                               HDCG_item / args.sample_times,
                                                                                               statistics.mean(
                                                                                                   HDCG_attribute_list),
                                                                                               total_reward / args.sample_times))
        print('Avg_Turn:{}\nAvg_REC_Turn:{}\nAvg_ASK_Turn:{}\nAvg_REC_STEP:{}\nAvg_ASK_STEP:{}'.format(
            AvgT / args.sample_times,
            len(rec_step_list) / args.sample_times,
            len(ask_step_list) / args.sample_times,
            statistics.mean(rec_step_list),
            statistics.mean(ask_step_list)))
        if epoch % args.eval_num == 0:
            SR15_mean = evaluate(args, kg, dataset, agent, filename, epoch)
            test_performance.append(SR15_mean)
        if epoch % args.save_num == 0:
            ask_agent.save_model(data_name=args.data_name, filename=filename, epoch_user=epoch)
            rec_agent.save_model(data_name=args.data_name, filename=filename, epoch_user=epoch)
            value_net.save_value_net(data_name=args.data_name, filename=filename, epoch_user=epoch)
    print(test_performance)


def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
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
    parser.add_argument('--seq', type=str, default='transformer', help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    parser.add_argument('--max_rec_step', type=int, default=10, help='max recommend step in one turn')
    parser.add_argument('--max_ask_step', type=int, default=3, help='max ask step in one turn')

    args = parser.parse_args()
    return args
