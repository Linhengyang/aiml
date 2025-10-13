# Optimizer for GPT
# AdamW

import torch
from torch.optim import Optimizer
from typing import Iterable, Optional, Dict, Any, Union


class AdamW(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)