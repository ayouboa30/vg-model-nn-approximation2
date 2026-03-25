import torch
from torch import nn
import torch.nn.functional as F
import math

class Linear(nn.Linear):
    def __init__(self, bias: bool = True, device = None, dtype = None) -> None:
        super().__init__(5, 1, bias=bias, device=device, dtype=dtype)
    
class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 3,
        device = None,
        dtype = None
    ) -> None:
        super().__init__()

        self.in_layer = nn.Linear(5, hidden_dim, device=device, dtype=dtype)
        self.hid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype) for _ in range(depth - 2)])
        self.out_layer = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = nn.functional.gelu(self.in_layer(x))

        for layer in self.hid_layers:
            x = nn.functional.gelu(layer(x))
        
        return self.out_layer(x)


class GeluMLP(nn.Module):
    """Modèle de référence classique avec GeLU"""
    def __init__(self, in_features: int = 5, hidden_dim: int = 128, depth: int = 4, device=None, dtype=None):
        super().__init__()
        self.in_layer = nn.Linear(in_features, hidden_dim, device=device, dtype=dtype)
        self.hid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype) for _ in range(depth - 2)])
        self.out_layer = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.in_layer(x))
        for layer in self.hid_layers:
            x = F.gelu(layer(x))
        return self.out_layer(x).squeeze(-1) 

class LogSpaceSoftplusMLP(nn.Module):
    """Modèle proposant g(x) = log(y) avec Softplus pour la lissité"""
    def __init__(self, in_features: int = 5, hidden_dim: int = 128, depth: int = 4, device=None, dtype=None):
        super().__init__()
        self.in_layer = nn.Linear(in_features, hidden_dim, device=device, dtype=dtype)
        self.hid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype) for _ in range(depth - 2)])
        self.out_layer = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = F.softplus(self.in_layer(x))
        for layer in self.hid_layers:
            x = F.softplus(layer(x))
        return self.out_layer(x).squeeze(-1) 
