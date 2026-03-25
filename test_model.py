import torch
import torch.nn.functional as F
from torch import nn

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.V = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.uniform_(self.V, -2.5, -1.0)
            nn.init.zeros_(self.bias)
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
            nn.init.uniform_(self.V, -2.5, -1.0)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        negative_weight = -F.softplus(self.V)
        return nn.functional.linear(x, negative_weight, self.bias)

class ConstrainedPricingModel(nn.Module):
    def __init__(self, hidden_dim=256, depth=5, device=None, dtype=None):
        super().__init__()

        self.u_layers = nn.ModuleList([
            nn.Linear(3 if i == 0 else hidden_dim, hidden_dim, device=device, dtype=dtype)
            for i in range(depth)
        ])

        self.k_layers = nn.ModuleList([
            NegativeLinear(1, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth)
        ])

        self.t_layers = nn.ModuleList([
            PositiveLinear(1, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth)
        ])

        self.z_layers = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim, device=device, dtype=dtype)
            for _ in range(depth - 1)
        ])

        self.out_layer_z = PositiveLinear(hidden_dim, 1, device=device, dtype=dtype)
        self.out_layer_k = NegativeLinear(1, 1, device=device, dtype=dtype)
        self.out_layer_t = PositiveLinear(1, 1, device=device, dtype=dtype)
        self.out_layer_u = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

        self.act_z = nn.Softplus()
        self.act_u = nn.GELU()

    def forward(self, inputs: torch.Tensor):
        t = inputs[:, 0:1]
        k = inputs[:, 1:2]
        u = inputs[:, 2:5]

        u_hiddens = []
        curr_u = u
        for u_layer in self.u_layers:
            curr_u = self.act_u(u_layer(curr_u))
            u_hiddens.append(curr_u)

        z = self.act_z(self.k_layers[0](k) + self.t_layers[0](t) + u_hiddens[0])
        for i in range(len(self.z_layers)):
            z_next = self.z_layers[i](z) + self.k_layers[i+1](k) + self.t_layers[i+1](t) + u_hiddens[i+1]
            z = self.act_z(z_next)

        output = self.out_layer_z(z) + self.out_layer_k(k) + self.out_layer_t(t) + self.out_layer_u(u_hiddens[-1])
        return F.softplus(output).squeeze(-1)

model = ConstrainedPricingModel()
x = torch.rand(10, 5, requires_grad=True)
y = model(x)

grad_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
grad_t = grad_x[:, 0]
grad_k = grad_x[:, 1]

print("Is monotonically increasing in T? (grad >= 0):", (grad_t >= 0).all().item())
print("Is monotonically decreasing in K? (grad <= 0):", (grad_k <= 0).all().item())

grad_k_k = torch.autograd.grad(grad_k.sum(), x, retain_graph=True)[0][:, 1]
print("Is convex in K? (2nd grad >= 0):", (grad_k_k >= 0).all().item())
