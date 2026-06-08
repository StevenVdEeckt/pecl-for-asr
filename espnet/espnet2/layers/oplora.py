import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

import espnet2.layers.loralib as lora

import logging

class OPLoRA(lora.LoRALayer):
    P_L: Optional[torch.Tensor]
    P_R: Optional[torch.Tensor]
    def __init__(
        self, 
        r: int,
        k: int,
        device: str,
        layer_name: str = "",
    ):
        self.r = r
        self.k = k
        self.A, self.B = None, None
        self.device = device
        self.layer_name = layer_name
        self.export_merge = False

    def _add_adapter(self):
        raise NotImplementedError("add_adapter must be overridden in the subclass")

    def reset_lora_parameters(self, lora_A, lora_B):
        raise NotImplementedError("reset_lora_parameters must be overridden in the subclass")

    def set_export_merge(self, b: bool):
        self.export_merge = b

class Linear(nn.Linear, OPLoRA):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0,
        k: int = 0,
        fan_in_fan_out: bool = False,
        adapt_weight: bool = False,
        layer_name: str = "",
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        OPLoRA.__init__(self, r=r, k=k, device=self.weight.device, layer_name=layer_name)
        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.weight.requires_grad = adapt_weight
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self._add_adapter()

    def _add_adapter(self):
        """Add a new adapter for the given language token and task label."""
        r = self.r
        if r > 0:
            self.A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
            self.B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))
            self.reset_lora_parameters(self.A, self.B)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    def update_adapter(self):
        # update is called when 'weight' is updated
        k = self.k
        if k > 0:
            U, E, Vh = torch.linalg.svd(self.weight.data, full_matrices=False)
            # set U_k, V_k
            U_k = U[:, :k]
            V_k = Vh[:k, ].T
            # compute projection matrix
            P_L = torch.eye(U_k.size(0), device=self.weight.device) - U_k @ U_k.T
            P_R = torch.eye(V_k.size(0), device=self.weight.device) - V_k @ V_k.T
            self.register_buffer("P_L", P_L, persistent=True)
            self.register_buffer("P_R", P_R, persistent=True)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def reset_lora_parameters(self, lora_A, lora_B):
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.zeros_(lora_B)

    def extra_repr(self):
        s = f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}; \n'
        """Extra representation of the module to include adapter information."""
        if hasattr(self, 'P_L'):
            s += f"OPLoRA: A={self.A.shape}, B={self.B.shape}, P_L={self.P_L}, P_R={self.P_R};"
        else:
            s += f"OPLoRA: A={self.A.shape}, B={self.B.shape};"
        return s

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to only store the reconstructed weight and bias."""
        state = nn.Linear.state_dict(self, destination, prefix, keep_vars)
        if not getattr(self, "export_merge", False):
            return state
        # Reconstruct the full weight matrix
        with torch.no_grad():
            full_weight = self.weight + self.P_L @ self.B @ self.A @ self.P_R
        # Replace entries in state_dict
        state[prefix + 'weight'] = full_weight
        # Remove U, S, Vh from state_dict
        state.pop(prefix + 'A', None)
        state.pop(prefix + 'B', None)
        state.pop(prefix + 'P_L', None)
        state.pop(prefix + 'P_R', None)
        return state

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        weight = self.P_L @ self.B @ self.A @ self.P_R
        result += F.linear(x, T(weight))
        return result