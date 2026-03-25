from typing import (
    Any,
    Optional,
    List,
    Tuple
)

import inspect

import torch
from torch import nn

# TODO : Very unclear code and pattern, need to rework that

class CombinedLoss(nn.Module):
    def __init__(self, losses: List[Tuple[nn.Module, float]]) -> None:
        super().__init__()
        self.any_requires_dx = False
        self.any_requires_hx = False

        signed_losses = []

        for (loss_fn, weight) in losses:
            if isinstance(loss_fn, PhysicsInformedLoss):
                signature = inspect.signature(loss_fn)
                requires_dx = "dx" in signature.parameters.keys()
                requires_hx = "hx" in signature.parameters.keys()
                signed_losses.append((loss_fn, weight, requires_dx, requires_hx))

                self.any_requires_dx = self.any_requires_dx or requires_dx
                self.any_requires_hx = self.any_requires_hx or requires_hx
            else:
                signed_losses.append((loss_fn, weight, False, False))

        self.any_requires_dx = self.any_requires_dx or self.any_requires_hx
        self.signed_losses = signed_losses

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, ic: torch.Tensor):

        if self.any_requires_dx:
            dx = torch.autograd.grad(
                outputs=y_hat,
                inputs=x,
                grad_outputs=torch.ones_like(y_hat, device=y_hat.device),
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            dx = None

        if self.any_requires_hx:
            hx = torch.autograd.grad(
                outputs=dx,
                inputs=x,
                grad_outputs=torch.ones_like(dx, device=dx.device),
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            hx = None

        loss = torch.tensor(0., device=y_hat.device)
        for (loss_fn, weight, requires_dx, requires_hx) in self.signed_losses:
            if isinstance(loss_fn, WeightedLoss):
                loss += weight * loss_fn(y_hat, y, ic)
            elif isinstance(loss_fn, PhysicsInformedLoss):
                kwargs = {}
                if requires_dx: kwargs["dx"] = dx
                if requires_hx: kwargs["hx"] = hx

                loss += weight * loss_fn(x, y_hat, **kwargs)
            else:
                loss += weight * loss_fn(y_hat, y)

        return loss
    
class WeightedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, ic: Optional[torch.Tensor] = None):
        raise NotImplementedError

class ThresholdedWeightedMSE(WeightedLoss):
    def __init__(self, precision: float = 1e-4) -> None:
        super().__init__()

        self.precision = precision

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, ic: Optional[torch.Tensor] = None):
        if ic is None: ic = torch.tensor(1., device=y.device)
        
        return torch.mean((y_hat - y) ** 2 / (ic + self.precision))

class PhysicsInformedLoss(nn.Module):
    def __init__(self, feature: int):
        super().__init__()
        self.feature = feature

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor):
        raise NotImplementedError

class MonotonyLoss(PhysicsInformedLoss):
    def __init__(self, feature: int, increasing: bool = True):
        super().__init__(feature)
        self.increasing = increasing

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor, *, dx: Optional[torch.Tensor] = None):
        if dx is None:
            dx = torch.autograd.grad(
                outputs=y_hat,
                inputs=x,
                grad_outputs=torch.ones_like(y_hat, device=y_hat.device),
                create_graph=True,
                retain_graph=True
            )[0]

            dx = dx[:, self.feature]

        return torch.mean((torch.clamp(dx, max=0.) if self.increasing else torch.clamp(dx, min=0.))**2)

class ConvexityLoss(PhysicsInformedLoss):
    def __init__(self, feature: int, convex: bool = True):
        super().__init__(feature)
        self.convex = convex

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor, *, hx: Optional[torch.Tensor] = None):
        if hx is None:
            dx = torch.autograd.grad(
                outputs=y_hat,
                inputs=x,
                grad_outputs=torch.ones_like(y_hat, device=y_hat.device),
                create_graph=True,
                retain_graph=True
            )[0]

            hx = torch.autograd.grad(
                outputs=dx,
                inputs=x,
                grad_outputs=torch.ones_like(dx, device=dx.device),
                create_graph=True,
                retain_graph=True
            )[0]

            hx = hx[:, self.feature]

        return torch.mean((torch.clamp(hx, max=0.) if self.convex else torch.clamp(hx, min=0.))**2)
    
