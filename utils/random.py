import random
import numpy as np
import torch

def setup_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank) 
    random.seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True