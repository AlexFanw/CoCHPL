import os
import torch

from rl.agent.ask_agent import FeatureDict
from rl.rl_option_critic import set_arguments, option_critic_pipeline
from utils.utils import load_kg, load_dataset

if __name__ == '__main__':
    # set arguments
    args = set_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    print('DEVICE: {}'.format(args.device))
    print('DATASET: {}'.format(args.data_name))

    # load dataset
    kg = load_kg(args.data_name)
    dataset = load_dataset(args.data_name)
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    args.attr_num = feature_length
    
    print('FEATURE NUMBER: {}'.format(feature_length))
    print('ATTRIBUTE NUMBER', args.attr_num)
    print('ENTROPY METHOD:', args.entropy_method)
    
    # train
    filename = 'train-datasets-{}-rl-cand_feature_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_feature_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    option_critic_pipeline(args, kg, dataset, filename)
