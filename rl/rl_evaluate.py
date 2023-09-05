import copy
import statistics
import time
from itertools import count
import math

import numpy as np
import torch

from utils.utils import *
from rl.recommend_env.env_variable_question import VariableRecommendEnv
from tqdm import tqdm

def choose_option(ask_agent, rec_agent, state, cand, option_strategy=0):
    if cand["feature"] == [] or len(cand["item"]) < 10:
        return 0  # Recommend
    with torch.no_grad():
        state_emb = ask_agent.gcn_net([state])
        feature_cand = cand["feature"]
        ask_score = []
        value = ask_agent.value_net(state_emb).detach().cpu().numpy().squeeze()
        for feature in feature_cand:
            feature = torch.LongTensor(np.array(feature).astype(int).reshape(-1, 1)).to(ask_agent.device)  # [N*1]
            feature = ask_agent.gcn_net.embedding(feature)
            ask_score.append(
                 value + ask_agent.policy_net(state_emb, feature, choose_action=False).detach().cpu().numpy().squeeze())
        ask_prop = np.exp(ask_score) / sum(np.exp(ask_score))
        if option_strategy == 0:
            ask_Q = np.array(ask_score).dot(ask_prop)
        else:
            ask_Q = max(ask_score)

        state_emb = rec_agent.gcn_net([state])
        item_cand = cand["item"]
        rec_score = []
        value = rec_agent.value_net(state_emb).detach().cpu().numpy().squeeze()
        for item in item_cand:
            item = torch.LongTensor(np.array(item).astype(int).reshape(-1, 1)).to(rec_agent.device)  # [N*1]
            item = rec_agent.gcn_net.embedding(item)
            rec_score.append(
                 value + rec_agent.policy_net(state_emb, item, choose_action=False).detach().cpu().numpy().squeeze())
        rec_prop = np.exp(rec_score) / sum(np.exp(rec_score))
        
        if option_strategy == 0:
            rec_Q = np.array(rec_score).dot(rec_prop)
        else:
            rec_Q = max(rec_score)
        print("\n**CHOOSE OPTION** ASK VALUE:{}, REC VALUE:{}\n".format(ask_Q, rec_Q))
        if ask_Q > rec_Q:
            return 1
        else:
            return 0
        
        
# def choose_option(ask_agent, rec_agent, state, cand):
#     if cand["feature"] == [] or len(cand["item"]) < 10:
#         return 0  # Recommend
#     with torch.no_grad():
#         state_emb = ask_agent.gcn_net([state])
#         feature_cand = cand["feature"]
#         ask_score = []
#         for feature in feature_cand:
#             feature = torch.LongTensor(np.array(feature).astype(int).reshape(-1, 1)).to(ask_agent.device)  # [N*1]
#             feature = ask_agent.gcn_net.embedding(feature)
#             # print(ask_agent.value_net(state_emb))
#             ask_score.append(
#                 ask_agent.value_net(state_emb).detach().cpu().numpy().squeeze() + ask_agent.policy_net(state_emb, feature,
#                                                                                                  choose_action=False).detach().cpu().numpy().squeeze())
#         # ask_Q = np.array(ask_score).dot(np.exp(ask_score) / sum(np.exp(ask_score)))
#         ask_Q = max(ask_score)

#         state_emb = rec_agent.gcn_net([state])
#         item_cand = cand["item"]
#         rec_score = []
#         for item in item_cand:
#             item = torch.LongTensor(np.array(item).astype(int).reshape(-1, 1)).to(rec_agent.device)  # [N*1]
#             item = rec_agent.gcn_net.embedding(item)
#             rec_score.append(
#                 rec_agent.value_net(state_emb).detach().cpu().numpy().squeeze() + rec_agent.policy_net(state_emb, item,
#                                                                                                  choose_action=False).detach().cpu().numpy().squeeze())
#         # rec_Q = np.array(rec_score).dot(np.exp(rec_score) / sum(np.exp(rec_score)))
#         rec_Q = max(rec_score)
#         # return ask_Q / (ask_Q + rec_Q), rec_Q / (ask_Q + rec_Q)
#         # return math.exp(ask_Q) / (math.exp(ask_Q) + math.exp(rec_Q)), math.exp(rec_Q) / (
#         #         math.exp(ask_Q) + math.exp(rec_Q))
#         if ask_Q > rec_Q:
#             return 1
#         else:
#             return 0


def infer_features(ask_agent, args, infer_env, infer_state, infer_cand, infer_action_space):
    blockPrint()
    done = 0
    termination = False
    chosen_features = []
    while not termination and not done:
        if infer_env.cur_conver_step > args.max_ask_step:
            break
        # Select Action
        chosen_feature = ask_agent.select_action(infer_state, infer_cand["feature"],
                                                 infer_action_space["feature"], is_test=True)
        chosen_features.append(chosen_feature)
        infer_reward = ask_agent.state_inferrer([infer_state], torch.LongTensor([[chosen_feature]]))
        infer_next_state, infer_next_cand, infer_action_space, reward, done = infer_env.step(
            attribute=chosen_feature.item(),
            items=None,
            mode="test",
            infer=infer_reward)
        if infer_next_state is None:
            break
        # Whether Termination
        infer_next_state_emb = ask_agent.gcn_net([infer_next_state])
        term_score = ask_agent.termination_net(infer_next_state_emb).item()
        print("Termination Score:", term_score)
        if term_score >= 0.5:
            termination = True
        if infer_next_cand["feature"] == []:
            termination = True

        if done:  # No cands
            break
        infer_state = infer_next_state
        infer_cand = infer_next_cand
    # enablePrint()
    return chosen_features


def infer_items(rec_agent, args, infer_env, infer_state, infer_cand, infer_action_space):
    blockPrint()
    done = 0
    termination = False
    chosen_items = []
    while not termination and not done:
        if infer_env.cur_conver_step > args.max_rec_step:
            break

        # Select Action
        chosen_item = rec_agent.select_action(infer_state, infer_cand["item"],
                                              infer_action_space["item"], is_test=True)
        chosen_items.append(chosen_item.item())
        infer_reward = rec_agent.state_inferrer([infer_state], torch.LongTensor([[chosen_item]]))
        infer_next_state, infer_next_cand, infer_action_space, reward, done = infer_env.step(
            attribute=None,
            items=chosen_items,
            mode="test",
            infer=infer_reward)
        if infer_next_state is None:
            break
        # Whether Termination
        infer_next_state_emb = rec_agent.gcn_net([infer_next_state])

        term_score = rec_agent.termination_net(infer_next_state_emb).item()
        print("Termination Score:", term_score)
        if term_score >= 0.5:
            termination = True
        if infer_next_cand["feature"] == []:
            termination = True

        if done:  # No cands / max turn
            break
        infer_state = infer_next_state
        infer_cand = infer_next_cand
    # enablePrint()
    return chosen_items


@torch.no_grad()
def rl_evaluate(args, kg, dataset, filename, epoch, ask_agent=None, rec_agent=None):
    tt = time.time()
    start = tt

    # Environment
    env = VariableRecommendEnv(kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                               cand_feature_num=args.cand_feature_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num,
                               mode='test', entropy_way=args.entropy_method)
    set_random_seed(args.seed)
    # Statistic initial
    AvgT_list = []
    Suc_Turn_list = []
    rec_step_list = []
    ask_step_list = []
    HDCG_item = 0.
    HDCG_attribute_list = []

    # Training/Test Size
    total_user_size = env.ui_array.shape[0]
    print('User size in UI_test: ', total_user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(epoch) + filename
    detail_filename = 'Detail-' + filename
    if args.eval_user_size == 0:
        user_size = total_user_size
    else:
        user_size = args.eval_user_size
    print('The select Test size : ', user_size)

    for user in tqdm(range(user_size)):
        blockPrint()
        print('\n================Episode:{}===================='.format(user))
        state, cand, action_space = env.reset()
        done = 0
        for t in range(1, 16):  # Turn
            chosen_features = []
            chosen_items = []
            '''
            Over Option : Select Ask / Rec
            '''
            env.cur_conver_step = 1
            if done:
                break
            print("Candidate: ", cand)
            option = choose_option(ask_agent, rec_agent, state, cand)

            '''
            Intra Option choose: Select features / items
            '''
            # ASK
            if option == 1:
                print("\n————————Turn: ", t, "  Option: ASK————————")
                ask_score = []
                # infer step
                infer_env = copy.deepcopy(env)
                infer_state = copy.deepcopy(state)
                infer_cand = copy.deepcopy(cand)
                infer_action_space = copy.deepcopy(action_space)
                chosen_features = infer_features(ask_agent, args,
                                                 infer_env, infer_state, infer_cand, infer_action_space)
                # interactive step
                for chosen_feature in chosen_features:

                    # Env Interaction
                    next_state, next_cand, action_space, reward, done = env.step(chosen_feature.item(), None)
                    ask_score.append(reward)

                    state = next_state
                    cand = next_cand

                    if done:
                        AvgT_list.append(t)
                        break

                # calculate HDCG Attribute
                for i in range(len(ask_score)):
                    if ask_score[i] > 0:
                        HDCG_attribute_list.append(
                            (1 / math.log(t + 2, 2) + (1 / math.log(t + 1, 2) - 1 / math.log(t + 2, 2)) /
                             math.log(i + 2, 2)))

            # RECOMMEND
            elif option == 0:
                print("\n————————Turn: ", t, "  Option: REC————————")

                # Infer Step
                infer_env = copy.deepcopy(env)
                infer_state = copy.deepcopy(state)
                infer_cand = copy.deepcopy(cand)
                infer_action_space = copy.deepcopy(action_space)

                chosen_items = infer_items(rec_agent, args,
                                           infer_env, infer_state, infer_cand, infer_action_space)

                # Env Interaction
                next_state, next_cand, action_space, reward, done = env.step(None, chosen_items, mode="test")
                state = next_state
                cand = next_cand
                if done:
                    if reward == env.reward_dict["rec_suc"]:  # recommend successfully
                        Suc_Turn_list.append(t)
                        HDCG_item = HDCG_item + (
                                1 / math.log(t + 2, 2) + (1 / math.log(t + 1, 2) - 1 / math.log(t + 2, 2)) /
                                math.log(done + 1, 2))

                    AvgT_list.append(t)

            if not done and t == env.max_turn:
                AvgT_list.append(t)

            if option == 1:
                ask_step_list.append(len(chosen_features))
            else:
                rec_step_list.append(len(chosen_items))

            env.cur_conver_turn += 1

    enablePrint()  # Enable print function
    print(len(AvgT_list), len(rec_step_list), len(ask_step_list))
    stl = np.array(Suc_Turn_list)
    AvgT = statistics.mean(AvgT_list)
    Avg_REC_Turn = len(rec_step_list) / user_size
    Avg_ASK_Turn = len(ask_step_list) / user_size
    Avg_REC_STEP = statistics.mean(rec_step_list) if len(rec_step_list) else 0
    Avg_ASK_STEP = statistics.mean(ask_step_list) if len(ask_step_list) else 0
    HDCG_item = HDCG_item / user_size
    HDCG_attribute = statistics.mean(HDCG_attribute_list) if len(HDCG_attribute_list) else 0
    SR = [0]
    for i in range(1, 16):
        SR.append(len(stl[stl <= i]) / user_size)
    print('\nSample Times:{}'.format(user_size))
    print('SR5:{}\n'
          'SR10:{}\n'
          'SR15:{}\n'
          'HDCG_item:{}\n'
          'HDCG_attribute:{}\n'.format(SR[5], SR[10], SR[15], HDCG_item, HDCG_attribute))
    print('Avg_Turn:{}\n'
          'Avg_REC_Turn:{}\n'
          'Avg_ASK_Turn:{}\n'
          'Avg_REC_STEP:{}\n'
          'Avg_ASK_STEP:{}'.format(AvgT, Avg_REC_Turn, Avg_ASK_Turn, Avg_REC_STEP, Avg_ASK_STEP))

    # SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean]

    results = [SR[5], SR[10], SR[15], AvgT, Avg_REC_Turn, Avg_ASK_Turn, Avg_REC_STEP, Avg_ASK_STEP, HDCG_item]
    # Single Epoch file
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=epoch, results=results,
                  spend_time=time.time() - start, mode='test')  # save rl SR

    PATH = CHECKPOINT_DIR[args.data_name] + '/log/' + detail_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('***EPOCH:{}***\n'.format(epoch))
        f.write('***User Size:{}***\n'.format(user_size))
        for i in range(len(SR)):
            f.write(">>>SR{}: {}\n".format(i, SR[i]))
        f.write('>>>HDCG_item: {}\n'.format(HDCG_item))
        f.write('>>>HDCG_attribute: {}\n'.format(HDCG_attribute))
        f.write('>>>Avg_Turn:{}\n'
                '>>>Avg_REC_Turn:{}\n'
                '>>>Avg_ASK_Turn:{}\n'
                '>>>Avg_REC_STEP:{}\n'
                '>>>Avg_ASK_STEP:{}\n\n'.format(AvgT, Avg_REC_Turn, Avg_ASK_Turn, Avg_REC_STEP, Avg_ASK_STEP))
    return results
