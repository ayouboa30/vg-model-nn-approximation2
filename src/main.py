from typing import (
    List,
    Dict,
    Optional,
)

import random
import torch
import numpy as np
from tqdm import tqdm

from cuda_vg import VGPricingDataset
from metrics import Loss
from models import ResidualMLP
from experiments import plot_model_evaluation, plot_learning_curves

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
):
    model.eval()
    
    with torch.no_grad():
        x, y = next(iter(loader))

        x, y = x.to(device), y.to(device)

        if hasattr(model, "log_forward"):
            y_hat = model.log_forward(x)
        else:
            y_hat = model(x)

        if hasattr(loss_fn, "log_forward"):
            loss = loss_fn.log_forward(y_hat, y)
        else:
            loss = loss_fn(y_hat, y)

    return loss.item()

class EarlyStopping:
    def __init__(self, patience: int = 7, monitor: str = "loss",  mode: str = "min", delta: float = 0.00001):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        if self.best_metric is None:
            self.best_metric = metrics[self.monitor]
            return False

        if self.mode == "min":
            improved = metrics[self.monitor] < (self.best_metric - self.delta)
        else:
            improved = metrics[self.monitor] > (self.best_metric + self.delta)

        if improved:
            self.best_metric = metrics[self.monitor]
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False

def main():
    seed = 1
    batch_size = 256
    epoch_size = 20
    max_epoch = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc_steps = 10_000
    param_priors = {
        "T": lambda size: np.random.uniform(low=0.1, high=2.0, size=size),
        "K": lambda size: np.full(size, 1.), # np.random.normal(loc=1, scale=0.001, size=size)
        "sigma": lambda size: np.random.uniform(low=0.05, high=0.6, size=size),
        "theta": lambda size: np.clip(np.random.normal(loc=-0.1, size=size), a_min=-0.5, a_max=0.2),
        "kappa": lambda size: np.exp(np.random.uniform(low=0.1, high=2.0, size=size)),
    }

    set_seed(seed)

    dataset = VGPricingDataset(**param_priors, mc_steps=mc_steps)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    loss_fn = Loss().to(device)
    # model = torch.nn.Linear(5, 2, bias=False).to(device)
    model = ResidualMLP().to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Learnable parameters : {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3,
        betas=(0.95, 0.99995), # Look back much further in the gradient history
    )

    scheduler = None
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-2, 
    #     steps_per_epoch=1,
    #     epochs=max_epoch,
    #     pct_start=0.1,
    #     anneal_strategy='cos'
    # )

    early_stopping = EarlyStopping(
        patience=50,
        monitor="loss",
        mode="min",
        delta=1e-5,
    )

    epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []

    while epoch < max_epoch:
        epoch += 1

        y_hat_batches = []
        y_batches = []

        model.train()
        for batch, (x, y) in enumerate(tqdm(loader, total=epoch_size, desc=f"Epoch {epoch}", postfix={
            "train_loss": train_losses[-1] if train_losses else "?" ,
            "val_loss": val_losses[-1] if val_losses else "?" ,
        }, leave=False)):
            if batch >= epoch_size:
                break

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if hasattr(model, "log_forward"):
                y_hat = model.log_forward(x)
            else:
                y_hat = model(x)

            if hasattr(loss_fn, "log_forward"):
                loss = loss_fn.log_forward(y_hat, y)
            else:
                loss = loss_fn(y_hat, y)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)
            
            optimizer.step()

            y_hat_batches.append(y_hat)
            y_batches.append(y)

        model.eval()
        with torch.no_grad():
            y_hat = torch.cat(y_hat_batches, dim=0)
            y = torch.cat(y_batches, dim=0)

            if hasattr(loss_fn, "log_forward"):
                loss = loss_fn.log_forward(y_hat, y)
            else:
                loss = loss_fn(y_hat, y)

            train_losses.append(loss.item())

        val_losses.append(evaluate(model, loss_fn, loader, device=device))

        learning_rates.append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

        if early_stopping(metrics={ "loss": val_losses[-1] }):
            print(f"Early stopping at epoch : {epoch}")
            break
    else:
        print(f"Hit max epoch : {epoch}")

    test_loss = evaluate(model, loss_fn, loader, device=device)
    print(f"Loss : {train_losses[-1]:.5f} (train) | {val_losses[-1]:.5f} (val) | {test_loss:.5f} (test)")

    print(f"Prior sampling time : {dataset.time_prior_sampling:.2f}s ({dataset.time_prior_sampling/dataset.samples:.8f}s/sample)")
    print(f"VG sampling time    : {dataset.time_vg_sampling:.2f}s ({dataset.time_vg_sampling/dataset.samples:.8f}s/sample)")

    plot_learning_curves(train_losses, val_losses, test_loss, learning_rates=learning_rates)

    plot_model_evaluation(
        dataset=dataset,
        model=model,
        parameter_labels=dataset.parameter_labels,
        parameter_ranges=[[0.08, 2.0], [0.9, 1.1], [0.05, 0.6], [-0.2, 0.], [np.exp(0.01), np.exp(2.)]], 
        parameter_values=[1., 1., 0.2, -0.1, np.exp(1.)],
        n=100,
    )

if __name__ == "__main__":
    main()