import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class ResidualMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 5,
        dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.encoder = nn.Linear(5, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.GELU(),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.GELU(),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(depth)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2),
        )

    def log_forward(self, x: torch.Tensor):

        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x) + x

        y = self.decoder(x)

        return y

    def forward(self, x: torch.Tensor):
        y = self.log_forward(x)
        y = torch.exp(y)

        return y
    