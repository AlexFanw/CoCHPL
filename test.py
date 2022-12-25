import numpy as np
import torch
from itertools import count
import random
if __name__ == "__main__":
    for i in range(100):
        soft_random = random.random()
        if soft_random >= 0.1:
            print(9)
        else:
            print(random.randint(0, 1))