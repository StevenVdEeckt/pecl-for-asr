
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


import logging


class SSVD_O:
    U: Optional[torch.Tensor]
    sigma: Optional[torch.Tensor]
    V: Optional[torch.Tensor]
    _ui: Optional[torch.Tensor]
    _uj: Optional[torch.Tensor]
    def __init__(
        self,
        p: float,
        device: str,
        layer_name: str = "",
        knowledge_preserved_mode: bool = False,
        rotation_only: bool = True,
    ):
        self.p = p
        self.delta_sigma, self.A = None, None
        self.G1 = None
        self.m, self.n, self.k = None, None, None
        self.device = device
        self.layer_name = layer_name
        self.std = 1e-4
        self.export_merge = False
        self.knowledge_preserved_mode = knowledge_preserved_mode
        self.rotation_only = rotation_only
        self._add_adapter()
        self.reset_lora_parameters(self.A, self.delta_sigma)

    def reset_lora_parameters(self, A, sigma):
        nn.init.normal_(A, mean=0.0, std=self.std)
        if sigma is not None:
            nn.init.normal_(sigma, mean=0.0, std=self.std)

    def _add_adapter(self):
        raise NotImplementedError("add_adapter must be overridden in the subclass")

    def set_export_merge(self, b: bool):
        self.export_merge = b

    def update_adapter(self):
        raise NotImplementedError("update must be overridden in the subclass")

    def _build_G(self):
        assert self.A is not None
        k = self.k
        n = self.n
        assert k <= n, f"Inner rank k={k} cannot exceed n={n}"
        K = torch.zeros(k, k, dtype=self.A.dtype, device=self.device)
        K[self._ui, self._uj] = self.A
        K[self._uj, self._ui] = -self.A
        I = torch.eye(k, dtype=self.A.dtype, device=self.device)
        Gk = I - 2 * K
        return Gk

    def _build_V(self):
        assert self.A is not None and self.V is not None
        k = self.k
        Gk = self._build_G()
        assert self.m is not None and self.n is not None and self.k is not None
        n = self.n
        assert k <= n, f"Inner rank k={k} cannot exceed n={n}"
        V = self.V
        V_prime = V.clone()
        if self.knowledge_preserved_mode:
            V_prime[:,-k:] = V[:,-k:] @ Gk.transpose(0, 1)
        else:
            V_prime[:, :k] = V[:, :k] @ Gk.transpose(0, 1)
        return V_prime

    def _build_sigma(self):
        assert self.sigma is not None and self.m is not None and self.n is not None
        if self.rotation_only:
            return self.sigma
        k = self.k
        n = self.sigma.numel()
        assert k <= n, f"Inner rank k={k} cannot exceed n={n}"
        delta_full = torch.zeros(n, dtype=self.sigma.dtype, device=self.device)
        if self.knowledge_preserved_mode:
            delta_full[-k:] = self.delta_sigma
        else:
            delta_full[:k] = self.delta_sigma
        return self.sigma + delta_full

    def _build_U(self):
        assert self.U is not None
        return self.U

    def _build_weight(self):
        return self._build_U() @ torch.diag(self._build_sigma()) @ torch.transpose(self._build_V(), 0, 1)


class Linear(nn.Linear, SSVD_O):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: float = 1.00,
        knowledge_preserved_mode: bool = False,
        fan_in_fan_out: bool = False,
        adapt_weight: bool = False,
        layer_name: str = "",
        rotation_only: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        SSVD_O.__init__(
            self,
            p=p,
            device=device,
            layer_name=layer_name,
            knowledge_preserved_mode=knowledge_preserved_mode,
            rotation_only=rotation_only,
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.weight.requires_grad = adapt_weight
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def _svd_weight(self):
        U, E, Vh = torch.linalg.svd(self.weight.data, full_matrices=False)
        V = Vh.transpose(0, 1)
        return U, E, V

    def _add_adapter(self):
        self.m, self.n = self.out_features, self.in_features
        d = min(self.out_features, self.in_features)
        U, E, V = self._svd_weight()
        k = int(self.p * d)
        if k < 1:
            raise ValueError(f"p={self.p} gives k=0 for SVD dimension d={d}; increase p.")
        self.k = k
        self.register_buffer("U", U, persistent=True)
        self.register_buffer("sigma", E, persistent=True)
        self.register_buffer("V", V.to(self.weight.device), persistent=True)
        param_shape = (int(k * (k - 1) / 2),)
        sigma_shape = (k,)
        if not self.rotation_only:
            self.delta_sigma = nn.Parameter(self.weight.new_zeros(sigma_shape))
            self.delta_sigma.requires_grad = True
        self.A = nn.Parameter(self.weight.new_zeros(param_shape))
        self.A.requires_grad = True
        idx = torch.triu_indices(k, k, offset=1, device=self.device)
        self.register_buffer("_ui", idx[0])
        self.register_buffer("_uj", idx[1])

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @torch.no_grad()
    def update_adapter(self):
        k = self.k
        n = self.n
        if (not self.knowledge_preserved_mode or torch.norm(self.A) / k < self.std) and self.G1 is None:
            self.update_adapter_old()
        else:
            logging.info(f"Updating the adapter (FROM SSVD)...")
            assert k <= n, f"Inner rank k={k} cannot exceed n={n}"
            G = self._build_G() if self.G1 is None else self.G1
            V_tail = self.V[:, -k:].clone()
            self.V[:, -k:].copy_(V_tail @ G.T)
            self.reset_lora_parameters(self.A, None)

    def update_adapter_old(self):
        logging.info(f"Updating the adapter (FROM W)...")
        U, E, V = self._svd_weight()
        with torch.no_grad():
            self.U.copy_(U)
            self.sigma.copy_(E)
            self.V.copy_(V.to(self.weight.device))
        logging.info(
            f"({self.layer_name}) Distance from SSVD-O to original W = "
            f"{torch.norm(self._build_weight() - self.weight.data).item():.3f} "
            f"(norm of W = {torch.norm(self.weight.data):.3f})"
        )

    def extra_repr(self):
        s = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}; \n"
        s += (
            f"SSVD: A={self.A.shape}, Sigma={None if self.rotation_only else self.delta_sigma.shape}, "
            f"k={self.k}, p={self.p}, Knowledge-preserved={self.knowledge_preserved_mode}, "
        )
        s += ";"
        return s

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = nn.Linear.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if not getattr(self, "export_merge", False):
            return state
        with torch.no_grad():
            W_merged = self._build_weight()
        state[prefix + "weight"] = W_merged
        for k in ["U", "sigma", "V", "delta_sigma", "A", "_ui", "_uj"]:
            state.pop(prefix + k, None)
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        g_key = prefix + "G"
        had_g = False
        if g_key in state_dict:
            self.G1 = state_dict.pop(g_key)
            had_g = True

        nn.Linear._load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

        if had_g:
            a_prefix = prefix + "A"
            missing_keys[:] = [
                k for k in missing_keys
                if not (k == a_prefix or k.startswith(a_prefix + "."))
            ]

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if getattr(self, "fan_in_fan_out", False) else w

        if getattr(self, "sigma", None) is None:
            return F.linear(x, T(self.weight), bias=self.bias)
        W_prime = self._build_weight()
        out = F.linear(x, T(W_prime), bias=self.bias)
        return out
