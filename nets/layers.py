import math
import torch
from torch import nn
import torch.nn.functional as F


class GTLayer(nn.Module):
    def __init__(self, **model_params):
        super(GTLayer, self).__init__()
        self.n_heads: int = model_params['n_heads']
        self.hidden_dim: int = model_params['hidden_dim']
        self.head_dim: int = self.hidden_dim // self.n_heads

        self.W_h = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=model_params['bias'])

        nn.init.orthogonal_(self.W_h.weight, gain=1.0)
        if model_params['bias']:
            nn.init.constant_(self.W_h.bias, 0.0)

        self.norm1 = Norm(**model_params)
        self.norm2 = Norm(**model_params)

        self.mlp = MLP(**model_params)

    def forward(self, h, e1, e2):
        batch_pomo_size, n_nodes, _ = h.shape
        h_in = h.clone()

        # Initial normalization
        h = self.norm1(h)

        # Linear transformation
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_pomo_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(batch_pomo_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(batch_pomo_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + e1
        att = F.softmax(att, dim=-1)
        att = att * e2
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Residual connection
        y = y.transpose(1, 2).reshape(batch_pomo_size, n_nodes, self.hidden_dim)
        y = y + h_in

        # Normalization and MLP
        out = self.mlp(self.norm2(y))

        # Final residual connection
        return out + y


class MLP(nn.Module):
    def __init__(self, d_in=None, d_out=None, mult=4, **model_params):
        super().__init__()
        if d_in is None:
            d_in = model_params['hidden_dim']
        if d_out is None:
            d_out = model_params['hidden_dim']

        self.act = Activation(model_params['activation'])
        bias = model_params['bias']
        dropout = model_params['dropout']
        d_hidden = int(round((mult * d_out)))

        if model_params['activation'] == "swiglu":
            self.c_fc = nn.Linear(d_in, 2 * d_hidden, bias=bias)  # gate+value
        else:
            self.c_fc = nn.Linear(d_in, d_hidden, bias=bias)

        self.c_proj = nn.Linear(d_hidden, d_out, bias=bias)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, device: torch.device | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.normalization = model_params['normalization']
        hidden_dim = model_params['hidden_dim']
        if self.normalization == 'layer':
            self.norm = nn.LayerNorm(hidden_dim)
        elif self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)
        elif self.normalization == 'rms':
            self.norm = RMSNorm(hidden_dim)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.normalization in ['instance', 'batch']:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        return x


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)
