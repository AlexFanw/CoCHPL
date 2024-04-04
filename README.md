# CoCHPL

🚀 The implementation of **Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation** (DASFAA 2024).

## 1. Training

```python
python train.py --data_name <data_name>
```

### 1.1 Optional arguments

For more detailed argument configuration, please refer to `RL/rl_option_critic.py`

```python
parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK, MOVIE])
parser.add_argument('--max_rec_step', type=int, default=10, help='max recommend step in one turn')
parser.add_argument('--max_ask_step', type=int, default=3, help='max ask step in one turn')
parser.add_argument('--cand_feature_num', type=int, default=10, help='candidate sampling number')
parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
```



## 2. Evaluation

```python
python evaluate.py --data_name <data_name> --load_rl_epoch <checkpoint_epoch> --eval_user_size 0
```

### 2.1 Optional arguments

For more detailed argument configuration, please refer to `RL/rl_option_critic.py`

```python
parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK, MOVIE])
parser.add_argument('--eval_user_size', '-eval_user_size', type=int, default=100, help='Select 0 to evaluate the full test dataset')
parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading rl model')
```

