import os
import random

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def remove_dataparallel_wrapper(state_dict):
    """
    Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary
    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = v

    return new_state_dict
