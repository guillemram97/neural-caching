import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F


EPS = 1e-12


class ModularModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._task_ids = None


class LoRALinear(ModularModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        r: int,
        lora_scaling: float,
        seed: int,
    ) -> None:
        super().__init__()

        if bias is None:
            bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        assert weight.size(0) == weight.size(1)

        D_MODEL = weight.size(0)

        self.r = r

        self.weight = nn.Parameter(weight.data, requires_grad=False)
        self.bias = nn.Parameter(bias.data, requires_grad=False)

        total_shape = (D_MODEL, r)

        self.A = nn.Parameter(torch.zeros(total_shape), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(total_shape), requires_grad=True)

        self.scaling = lora_scaling
        self.reset_parameters()

    def reset_parameters(self):
        # We init in such a way to have A*B equal to 0 at first
        # It's crucial for convergence
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, input: Tensor) -> Tensor:
        """
        input: [batch_size, seq_length, input_features]
        """

        BATCH, SEQ_LEN, D_MODEL = input.shape

        AB = torch.einsum("ir,or->io", self.A, self.B)

        output = torch.einsum("bni,io->bno", input, AB)
        output = F.linear(input, self.weight, self.bias) + output * self.scaling
        return output
