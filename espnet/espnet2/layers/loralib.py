import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import logging

class LoRALayer:
    def __init__(
        self, 
        r: int,
        lora_alpha: int, 
        lora_dropout: float,
        device: str,
        layer_name: str = "",
    ):
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.r = r
        self.lora_A, self.lora_B, self.scaling = None, None, None
        self.device = device
        self.layer_name = layer_name

    def reset_lora_parameters(self, lora_A, lora_B):
        raise NotImplementedError("reset_lora_parameters must be overridden in the subclass")

    def _add_adapter(self):
        raise NotImplementedError("add_adapter must be overridden in the subclass")

class Linear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        adapt_weight: bool = False,
        layer_name: str = "",
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                device=self.weight.device, layer_name=layer_name)
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
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))
            self.scaling = self.lora_alpha / r
            self.reset_lora_parameters(self.lora_A, self.lora_B)
            if self.bias is not None:
                self.bias.requires_grad_(False)

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
        s += f"lora: {self.lora_A.shape}, {self.lora_B.shape}, scaling={self.scaling};"
        return s

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to only store the reconstructed weight and bias."""
        state = nn.Linear.state_dict(self, destination, prefix, keep_vars)
        # Reconstruct the full weight matrix
        full_weight = self.weight + self.lora_B @ self.lora_A * self.scaling
        # Replace entries in state_dict
        state[prefix + 'weight'] = full_weight
        # Remove U, S, Vh from state_dict
        state.pop(prefix + 'lora_A', None)
        state.pop(prefix + 'lora_B', None)
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
        # Call the original _load_from_state_dict to load weights
        nn.Linear._load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # After loading the weights, set W_old to the current weights
        #logging.info(f"Are missing keys {[self.layer_name + '.lora_A', self.layer_name + '.lora_B']} in missing keys {missing_keys}?")
        for key in [self.layer_name + ".lora_A", self.layer_name + ".lora_B"]:
            if key in missing_keys:
                missing_keys.remove(key)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        weight = self.lora_B @ self.lora_A * self.scaling
        result += F.linear(self.lora_dropout(x), T(weight))
        return result
