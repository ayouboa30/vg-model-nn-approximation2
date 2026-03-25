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
        return nn.functional.linear(x, F.softplus(self.V), self.bias)

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
        return nn.functional.linear(x, -F.softplus(self.V), self.bias)

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
        self.act_t = nn.Tanh()  # Tanh is strictly increasing but allows concavity
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
        return output.squeeze(-1) # F.softplus is increasing but not negative convex, so we can return directly or use softplus for price positivity

model = ConstrainedPricingModel()
x = torch.rand(100, 5, requires_grad=True)
y = model(x)

grad_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
grad_t = grad_x[:, 0]
grad_k = grad_x[:, 1]

print("Is monotonically increasing in T? (grad >= 0):", (grad_t >= -1e-6).all().item())
print("Is monotonically decreasing in K? (grad <= 0):", (grad_k <= 1e-6).all().item())

grad_k_k = torch.autograd.grad(grad_k.sum(), x, retain_graph=True)[0][:, 1]
print("Is convex in K? (2nd grad >= 0):", (grad_k_k >= -1e-6).all().item())

grad_t_t = torch.autograd.grad(grad_t.sum(), x, retain_graph=True)[0][:, 0]
print("Is convex in T? (We DON'T want this strictly enforced):", (grad_t_t >= -1e-6).all().item())
