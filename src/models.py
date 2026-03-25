import torch
from torch import nn
import torch.nn.functional as F
import math

class Linear(nn.Linear):
    def __init__(self, bias: bool = True, device = None, dtype = None) -> None:
        super().__init__(5, 1, bias=bias, device=device, dtype=dtype)


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.V = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():

            nn.init.uniform_(self.V, -4.0, -2.0)
            fan_in = self.V.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.1
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):

        positive_weight = F.softplus(self.V)
        return nn.functional.linear(x, positive_weight, self.bias)

class NegativeLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.V = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():

            nn.init.uniform_(self.V, -4.0, -2.0)
            fan_in = self.V.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.1
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        negative_weight = -F.softplus(self.V)
        return nn.functional.linear(x, negative_weight, self.bias)
    
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





class PICNN(nn.Module):
    def __init__(self, hidden_dim=256, depth=5, device=None, dtype=None):
        super().__init__()

        self.u_layers = nn.ModuleList([
            nn.Linear(4 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])

        self.x_layers = nn.ModuleList([
            nn.Linear(1, hidden_dim, device=device, dtype=dtype) 
            for _ in range(depth)
        ])

        self.z_layers = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim, device=device, dtype=dtype) 
            for _ in range(depth - 1)
        ])

        self.out_layer_z = PositiveLinear(hidden_dim, 1, device=device, dtype=dtype)
        self.out_layer_x = nn.Linear(1, 1, device=device, dtype=dtype)
        self.out_layer_u = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

        self.act_z = nn.Softplus()
        self.act_u = nn.GELU()     

    def forward(self, inputs: torch.Tensor):
        x_c = inputs[:, 1:2] 
        
        u = inputs[:, [0, 2, 3, 4]] 
        u_hiddens = []
        curr_u = u
        for u_layer in self.u_layers:
            curr_u = self.act_u(u_layer(curr_u))
            u_hiddens.append(curr_u)
        z = self.act_z(self.x_layers[0](x_c) + u_hiddens[0])
        for i in range(len(self.z_layers)):
            z_next = self.z_layers[i](z) + self.x_layers[i+1](x_c) + u_hiddens[i+1]
            z = self.act_z(z_next)
        output = self.out_layer_z(z) + self.out_layer_x(x_c) + self.out_layer_u(u_hiddens[-1])
        return F.softplus(output).squeeze(-1)

class ConstrainedPricingModel(nn.Module):
    def __init__(self, hidden_dim=64, depth=4, device=None, dtype=None):
        super().__init__()

        # U-path (unconstrained)
        self.u_layers = nn.ModuleList([
            nn.Linear(3 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])

        # T-path (increasing in T)
        self.t_layers_T = nn.ModuleList([
            PositiveLinear(1 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])
        self.t_layers_U = nn.ModuleList([
            nn.Linear(3 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])

        # Z-path (convex and decreasing in K)
        self.z_layers_Z = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth - 1)
        ])
        self.z_layers_K = nn.ModuleList([
            NegativeLinear(1, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth)
        ])
        self.z_layers_T = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth)
        ])
        self.z_layers_U = nn.ModuleList([
            nn.Linear(3 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])

        self.out_Z = PositiveLinear(hidden_dim, 1, device=device, dtype=dtype)
        self.out_K = NegativeLinear(1, 1, device=device, dtype=dtype)
        self.out_T = PositiveLinear(hidden_dim, 1, device=device, dtype=dtype)
        self.out_U = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

        self.act_u = nn.GELU()
        self.act_t = nn.Softplus()  # Tanh is strictly increasing but allows concavity
        self.act_z = nn.Softplus() # Softplus is convex and increasing

    def forward(self, inputs: torch.Tensor):
        t = inputs[:, 0:1]
        k = inputs[:, 1:2]
        u = inputs[:, 2:5]

        # Initial states
        curr_u = u
        curr_t = t

        # We need to collect u_hiddens and t_hiddens
        u_hiddens = []
        t_hiddens = []

        for i in range(len(self.u_layers)):
            curr_u_next = self.act_u(self.u_layers[i](curr_u))
            curr_t_next = self.act_t(self.t_layers_T[i](curr_t) + self.t_layers_U[i](curr_u))

            u_hiddens.append(curr_u_next)
            t_hiddens.append(curr_t_next)

            curr_u = curr_u_next
            curr_t = curr_t_next

        z = self.act_z(self.z_layers_K[0](k) + self.z_layers_T[0](t_hiddens[0]) + self.z_layers_U[0](u))

        for i in range(len(self.z_layers_Z)):
            z_next = self.z_layers_Z[i](z) + self.z_layers_K[i+1](k) + self.z_layers_T[i+1](t_hiddens[i+1]) + self.z_layers_U[i+1](u_hiddens[i])
            z = self.act_z(z_next)

        output = self.out_Z(z) + self.out_K(k) + self.out_T(t_hiddens[-1]) + self.out_U(u_hiddens[-1])
        return F.softplus(output).squeeze(-1) # F.softplus is increasing but not negative convex, so we can return directly or use softplus for price positivity
