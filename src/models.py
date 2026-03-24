import torch
from torch import nn

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
        x = nn.functional.softplus(self.in_layer(x))

        for layer in self.hid_layers:
            x = nn.functional.softplus(layer(x))
        
        return nn.functional.softplus(self.out_layer(x))


class ICNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 3,
        device = None,
        dtype = None
    ) -> None:
        super().__init__()

        self.z_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype) 
            for _ in range(depth - 1)
        ])
        
        self.x_layers = nn.ModuleList([
            nn.Linear(5, hidden_dim, device=device, dtype=dtype) 
            for _ in range(depth)
        ])

        self.out_layer = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        z = nn.functional.softplus(self.x_layers[0](x))
        
        for i in range(len(self.z_layers)):
            z = nn.functional.softplus(self.z_layers[i](z) + self.x_layers[i+1](x))
            
        return self.out_layer(z)
