

from rl.rl_evaluate import rl_evaluate
from rl.rl_memory import ReplayMemoryPER
from rl.rl_option_critic import set_arguments
from rl.agent.ask_agent import AskAgent
from rl.agent.rec_agent import RecAgent
from rl.network.network_value import ValueNetwork
from rl.recommend_env.env_variable_question import VariableRecommendEnv
from utils.utils import *

# TODO select env
from graph.gcn import GraphEncoder
import warnings

warnings.filterwarnings("ignore")
RecommendEnv = {
    LAST_FM_STAR: VariableRecommendEnv,
    YELP_STAR: VariableRecommendEnv
}

FeatureDict = {
    LAST_FM_STAR: 'feature',
    YELP_STAR: 'feature',
    BOOK:'feature',
    MOVIE:'feature'
}


def evaluate(args, kg, dataset, filename):
    """rl Model Train

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
                               cand_feature_num=args.cand_feature_num, cand_item_num=args.cand_item_num,
                               attr_num=args.attr_num, mode='test',
                               entropy_way=args.entropy_method)

    # User&Feature Embedding
    embed = torch.FloatTensor(
        np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1, env.ui_embeds.shape[1]))), axis=0))
    # print(embed.size(0), embed.size(1))
    '''
    VALUE NET
    '''
    value_net = ValueNetwork().to(args.device)
    gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
                           embeddings=embed, fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn,
                           hidden_size=args.hidden_size).to(args.device)
    '''
    ASK AGENT
    '''
    # # ASK GCN Transformer
    # ask_gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
    #                            embeddings=embed, fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn,
    #                            hidden_size=args.hidden_size).to(args.device)
    # Ask Memory
    ask_memory = ReplayMemoryPER(args.memory_size)  # 50000
    # Ask Agent
    ask_agent = AskAgent(device=args.device, memory=ask_memory, action_size=embed.size(1),
                         hidden_size=args.hidden_size, gcn_net=gcn_net, learning_rate=args.learning_rate,
                         l2_norm=args.l2_norm, PADDING_ID=embed.size(0) - 1, value_net=value_net)
    '''
    REC AGENT
    '''
    # # Rec GCN Transformer
    # rec_gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg,
    #                            embeddings=embed, fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn,
    #                            hidden_size=args.hidden_size).to(args.device)
    # Rec Memory
    rec_memory = ReplayMemoryPER(args.memory_size)  # 50000
    # Rec Agent
    rec_agent = RecAgent(device=args.device, memory=rec_memory, action_size=embed.size(1),
                         hidden_size=args.hidden_size, gcn_net=gcn_net, learning_rate=args.learning_rate,
                         l2_norm=args.l2_norm, PADDING_ID=embed.size(0) - 1, value_net=value_net)
    # load parameters
    print('Loading Model in epoch {}'.format(args.load_rl_epoch))
    ask_agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    rec_agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    value_net.load_value_net(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    _ = rl_evaluate(args, kg, dataset, filename, args.load_rl_epoch, ask_agent, rec_agent)


if __name__ == '__main__':
    # Set arguments
    args = set_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('DEVICE: {}'.format(args.device))
    print('DATASET: {}'.format(args.data_name))

    # Load dataset
    kg = load_kg(args.data_name)
    dataset = load_dataset(args.data_name)
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    args.attr_num = feature_length
    
    print('FEATURE NUMBER: {}'.format(feature_length))
    print('ATTRIBUTE NUMBER', args.attr_num)
    print('ENTROPY METHOD:', args.entropy_method)

    # Evaluate
    filename = 'train-datasets-{}-rl-cand_feature_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_feature_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    evaluate(args, kg, dataset, filename)