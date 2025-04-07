import random
import numpy as np
import torch

def set_seed(seed: int = 42, 
             cudnn_deterministic: bool = True,
             cudnn_benchmark: bool = False
             ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark