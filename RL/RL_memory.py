from collections import namedtuple
from utils.utils import *
from RL.RL_sumtree import SumTree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand_items', 'next_cand_features'))


class ReplayMemoryPER(object):
    """Prioritized Experience Replay

    """
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        batch_data = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            batch_data.append(data)
            priorities.append(p)
            idxs.append(idx)

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return idxs, batch_data, is_weight

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
