import time
from itertools import count
import math
from utils.utils import *
from RL.recommend_env.env_variable_question import VariableRecommendEnv
from tqdm import tqdm


def evaluate(args, kg, dataset, filename, i_episode, ask_agent=None, rec_agent=None, value_net=None):
    env = VariableRecommendEnv(kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                    cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num,
                                    mode='test', entropy_way=args.entropy_method)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt

    SR5, SR10, SR15, AvgT, HDCG_item, total_reward = 0., 0., 0., 0., 0., 0.
    rec_step_list = []
    ask_step_list = []
    HDCG_attribute_list = []
    SR_turn_15 = [0] * args.max_turn
    turn_result = []
    result = []
    total_user_size = env.ui_array.shape[0]
    print('User size in UI_test: ', total_user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    plot_filename = 'Evaluate-'.format(i_episode) + filename
    user_size = 3000
    print('The select Test size : ', user_size)
    for user_num in tqdm(range(user_size)):  # user_size
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

        if (user_num + 1) % args.observe_num == 0 and user_num > 0:
            SR = [SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num, AvgT / args.observe_num,
                  Rank / args.observe_num, total_reward / args.observe_num]
            SR_TURN = [i / args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                               AvgT / args.observe_num, Rank / args.observe_num,
                                               total_reward / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        enablePrint()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    reward_mean = np.mean(np.array([item[5] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all,
                  spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean,
                                                                         Rank_mean, reward_mean))
    PATH = CHECKPOINT_DIR[args.data_name] + '/log/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epoch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    PATH = CHECKPOINT_DIR[args.data_name] + '/log/' + plot_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(i_episode, SR15_mean, AvgT_mean, Rank_mean, reward_mean))

    return SR15_mean
