import os
import torch
from collections import OrderedDict

def get_paras_num(model):
    paras =  sum(p.numel() for p in model.parameters())
    # print('paras num', paras)
    return paras

def remove_prefix_from_state_dict(state_dict, j: int = 1):
    new_state_dict = OrderedDict()
    for k, _ in state_dict.items():
        tokens = k.split(".")
        new_state_dict[".".join(tokens[j:])] = state_dict[k]

    return new_state_dict

