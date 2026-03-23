import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, delta: float = 0.5) -> None:
        super().__init__()

        self.delta = delta

    def log_forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        y = torch.clamp(y, min=1e-7)
        y = torch.log(y)

        return self.forward(y_hat, y)
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):

        return torch.mean(self.delta ** 2 * (torch.sqrt(1 + ((y_hat - y) / self.delta)**2) - 1))
