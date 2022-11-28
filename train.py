import os
import torch

from RL.agent.agent import FeatureDict
from RL.RL_option_critic import set_arguments, option_critic_train
from utils.utils import load_kg, load_dataset


if __name__ == '__main__':
    # Set arguments
    args = set_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('DEVICE: {}'.format(args.device))
    print('DATASET: {}'.format(args.data_name))

    # Load kg.pkl
    kg = load_kg(args.data_name)
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('FEATURE NUMBER: {}'.format(feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('ATTRIBUTE NUMBER', args.attr_num)
    print('ENTROPY METHOD:', args.entropy_method)

    # Load dataset.pkl
    dataset = load_dataset(args.data_name)

    # Train
    filename = 'train-datasets-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    # train(args, kg, dataset, filename)
    option_critic_train(args, kg, dataset, filename)
