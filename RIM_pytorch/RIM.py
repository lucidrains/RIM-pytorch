import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class RIM(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
