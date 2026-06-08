import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
import random

import espnet2.layers.loralib as lora
import logging


class BiLoRA(lora.LoRALayer):
    idx: Optional[torch.Tensor]
    idx_pair: Optional[torch.Tensor]
    def __init__(
        self, 
        r: int,
        device: str,
        layer_name: str = "",
    ):
        self.r = r
        self.Omega_real, self.Omega_img = None, None
        self.device = device
        self.layer_name = layer_name
        self.export_merge = False
        self._add_adapter()

    def _add_adapter(self):
        raise NotImplementedError("add_adapter must be overridden in the subclass")

    @staticmethod
    def sample_pairs_with_replacement(d0, d1, k_pairs, device):
        # Avoid self-pairs: (u,v) == (-u mod d0, -v mod d1)
        idx = torch.empty((k_pairs, 2), dtype=torch.long, device=device)
        i = 0
        while i < k_pairs:
            u = torch.randint(0, d0, (1,), device=device).item()
            v = torch.randint(0, d1, (1,), device=device).item()
            u2 = (-u) % d0
            v2 = (-v) % d1
            if u == u2 and v == v2:
                continue  # self-pair, skip
            idx[i, 0] = u
            idx[i, 1] = v
            i += 1
        return idx

    def reset_lora_parameters(self, lora_A, lora_B):
        raise NotImplementedError("reset_lora_parameters must be overridden in the subclass")

    def set_export_merge(self, b: bool):
        self.export_merge = b

    def _build_delta(self):
        d0, d1 = self.out_features, self.in_features
        B = torch.zeros((d0, d1), device=self.weight.device, dtype=torch.complex64)
        omega = self.Omega_real.to(B.dtype) + 1j * self.Omega_img.to(B.dtype)
        B[self.idx[:, 0], self.idx[:, 1]] = omega
        B[self.idx_pair[:, 0], self.idx_pair[:, 1]] = torch.conj(omega)
        return torch.fft.ifft2(B, norm="ortho").real.to(self.weight.dtype)

class Linear(nn.Linear, BiLoRA):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0,
        fan_in_fan_out: bool = False,
        adapt_weight: bool = False,
        layer_name: str = "",
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BiLoRA.__init__(self, r=r, device=self.weight.device, layer_name=layer_name)
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
        """Add BiLoRA adapter: sample k_pairs frequency pairs (with replacement) and
        create Omega (one complex value per pair)."""
        # number of complex trainable coefficients (match LoRA trainable params)
        self.k_pairs = (self.out_features * self.r + self.r * self.in_features) // 2
        k_pairs = self.k_pairs
        self.k = k_pairs * 2
        if k_pairs <= 0:
            return
        d0, d1 = self.out_features, self.in_features
        # sample primary indices; compute their conjugate partners
        for i in range(2):
            idx = self.sample_pairs_with_replacement(d0, d1, k_pairs, device=self.weight.device)
        # store indices as buffers (not masks)
        self.register_buffer("idx", idx, persistent=True)  # (k_pairs, 2)
        idx_pair = torch.stack(
            [(-idx[:, 0]) % d0, (-idx[:, 1]) % d1],
            dim=1
        )
        self.register_buffer("idx_pair", idx_pair, persistent=True)
        # trainable complex coefficients (one per pair)
        self.Omega_real = nn.Parameter(self.weight.new_zeros(k_pairs))
        self.Omega_img = nn.Parameter(self.weight.new_zeros(k_pairs))

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def extra_repr(self):
        s = f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}; \n'
        """Extra representation of the module to include adapter information."""
        s += (f"BiLoRA: k={self.k}, Omega={self.Omega_real.shape if self.Omega_real is not None else None}, "
              f"Omega-img={self.Omega_img.shape if self.Omega_img is not None else None};")
        return s

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to only store the reconstructed weight and bias."""
        state = nn.Linear.state_dict(self, destination, prefix, keep_vars)
        if not getattr(self, "export_merge", False):
            return state
        # Reconstruct the full weight matrix
        with torch.no_grad():
            full_weight = self.weight + self._build_delta()
        # Replace entries in state_dict
        state[prefix + 'weight'] = full_weight
        # Remove U, S, Vh from state_dict
        state.pop(prefix + 'idx', None)
        state.pop(prefix + 'idx_pair', None)
        state.pop(prefix + 'Omega_real', None)
        state.pop(prefix + 'Omega_img', None)
        return state

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        weight = self._build_delta()
        result += F.linear(x, T(weight))
        return result