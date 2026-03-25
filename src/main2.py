import random
from typing import Dict, Optional, Tuple, List

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from cuda_vg import VGPricingDataset
from experiments import plot_model_evaluation

from metrics import (
    CombinedLoss,
    ThresholdedWeightedMSE,
    MonotonyLoss,
    ConvexityLoss,
    ExpThresholdedWeightedMSE,
    LogMonotonyLoss,
    LogConvexityLoss
)

from models import MLP, LogSpaceSoftplusMLP

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
    model: torch.nn.Module,
    loss_fn: CombinedLoss,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
):
    model.eval()
    
    x, y, ic = next(iter(loader))
    x, y, ic = x.to(device), y.to(device), ic.to(device)
    x.requires_grad_()

    y_hat = model(x)
    if y_hat.dim() > 1 and y_hat.shape[-1] == 1:
        y_hat = y_hat.squeeze(-1)

    loss = loss_fn(x, y_hat, y, ic)
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

def train_model_experiment(
    model: torch.nn.Module, 
    loss_fn: CombinedLoss, 
    loader: torch.utils.data.DataLoader, 
    device: str, 
    max_epoch: int, 
    epoch_size: int,
    experiment_name: str
) -> Tuple[List[float], List[float], float]:
    """Entraîne un modèle et retourne l'historique des pertes ainsi que la perte sur le test set."""
    print(f"\n{'='*50}")
    print(f"DÉMARRAGE DE L'ENTRAÎNEMENT : {experiment_name}")
    print(f"{'='*50}")
    print(f"Modèle : {model.__class__.__name__}")
    print(f"Paramètres : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
         optimizer,
         max_lr=5e-3, 
         steps_per_epoch=1,
         epochs=max_epoch,
         pct_start=0.3,
         anneal_strategy="cos"
     )

    early_stopping = EarlyStopping(
        patience=50,
        monitor="loss",
        mode="min",
        delta=1e-5,
    )

    epoch = 0
    train_losses = []
    val_losses = []

    while epoch < max_epoch:
        epoch += 1
        epoch_train_losses = []

        model.train()
        for batch, (x, y, ic) in enumerate(tqdm(loader, total=epoch_size, desc=f"Epoch {epoch}", postfix={
            "train_loss": f"{train_losses[-1]:.5f}" if train_losses else "?",
            "val_loss": f"{val_losses[-1]:.5f}" if val_losses else "?",
        }, leave=False)):
            if batch >= epoch_size:
                break

            x, y, ic = x.to(device), y.to(device), ic.to(device)
            x.requires_grad_()

            optimizer.zero_grad()
            y_hat = model(x)
            if y_hat.dim() > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
                
            loss = loss_fn(x, y_hat, y, ic)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_train_losses.append(loss.item())

        train_losses.append(torch.mean(torch.tensor(epoch_train_losses)).item())
        val_losses.append(evaluate(model, loss_fn, loader, device=device))

        scheduler.step()

        if early_stopping(metrics={ "loss": val_losses[-1] }):
            print(f"Early stopping déclenché à l'epoch : {epoch}")
            break

    test_loss = evaluate(model, loss_fn, loader, device=device)
    print(f"Fin de l'entraînement. Loss : {train_losses[-1]:.5f} (train) | {val_losses[-1]:.5f} (val) | {test_loss:.5f} (test)")
    
    return train_losses, val_losses, test_loss

def main():
    seed = 1
    batch_size = 256
    epoch_size = 20
    max_epoch = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc_steps = 32_768

    param_priors = {
        "T": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0),
        "K": lambda size: torch.empty((size,), device=device).uniform_(0.8, 1.2), 
        "sigma": lambda size: torch.empty((size,), device=device).uniform_(0.05, 0.6),
        "theta": lambda size: torch.empty((size,), device=device).uniform_(-0.5, 0.05), 
        "kappa": lambda size: torch.empty((size,), device=device).uniform_(0.1, 1.0), 
    }

    set_seed(seed)

    print("Génération du dataset...")
    dataset = VGPricingDataset(**param_priors, mc_steps=mc_steps)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    model_mlp = MLP(hidden_dim=64, depth=4, device=device)
    loss_fn_mlp = CombinedLoss([
        (ThresholdedWeightedMSE(precision=1e-8), 1.),
        (MonotonyLoss(1, increasing=False), 1.), # K décroissant
        (MonotonyLoss(0, increasing=True), 1.),  # T croissant
        (ConvexityLoss(1, convex=True), 1.),     # Convexe en K
    ])
    
    train_mlp, val_mlp, test_mlp = train_model_experiment(
        model=model_mlp, 
        loss_fn=loss_fn_mlp, 
        loader=loader, 
        device=device, 
        max_epoch=max_epoch, 
        epoch_size=epoch_size,
        experiment_name="MLP Original (Softplus dans l'espace des prix)"
    )


    model_log = LogSpaceSoftplusMLP(hidden_dim=64, depth=4, device=device)
    loss_fn_log = CombinedLoss([
        (ExpThresholdedWeightedMSE(precision=1e-8), 1.), 
        (LogMonotonyLoss(1, increasing=False), 1.),
        (LogMonotonyLoss(0, increasing=True), 1.),
        (LogConvexityLoss(1, convex=True), 1.),
    ])
    
    train_log, val_log, test_log = train_model_experiment(
        model=model_log, 
        loss_fn=loss_fn_log, 
        loader=loader, 
        device=device, 
        max_epoch=max_epoch, 
        epoch_size=epoch_size,
        experiment_name="LogSpace Softplus MLP (Espace Log)"
    )

    # ==========================================
    # COMPARAISON ET VISUALISATION
    # ==========================================
    print(f"\n{'='*50}")
    print("RÉSUMÉ DES PERFORMANCES (Test Loss)")
    print(f"MLP Original     : {test_mlp:.7f}")
    print(f"LogSpace Softplus: {test_log:.7f}")
    print(f"{'='*50}\n")

    # 1. Plot des courbes d'apprentissage comparées
    plt.figure(figsize=(12, 7))
    plt.plot(train_mlp, label="MLP Original - Train", linestyle=':', color='#1f77b4', linewidth=2)
    plt.plot(val_mlp, label="MLP Original - Val", linestyle='-', color='#1f77b4', linewidth=2)
    
    plt.plot(train_log, label="LogSpace - Train", linestyle=':', color='#2ca02c', linewidth=2)
    plt.plot(val_log, label="LogSpace - Val", linestyle='-', color='#2ca02c', linewidth=2)

    plt.yscale("log")
    plt.xlabel("Époques", fontsize=12)
    plt.ylabel("Combined Loss (Log Scale)", fontsize=12)
    plt.title("Comparaison de convergence : MLP Original vs LogSpace", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    eval_ranges = [[0.08, 2.0], [0.9, 1.1], [0.05, 0.6], [-0.2, 0.], [np.exp(0.01), np.exp(2.)]]
    eval_values = [1., 1., 0.2, -0.1, np.exp(1.)]

    print("Génération des graphiques d'évaluation pour le MLP Original...")
    
    class SqueezeWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            y_hat = self.base_model(x)
            return y_hat.squeeze(-1) if y_hat.dim() > 1 else y_hat
            
    wrapped_mlp_model = SqueezeWrapper(model_mlp)

    plot_model_evaluation(
        dataset=dataset,
        model=wrapped_mlp_model,
        parameter_labels=dataset.parameter_labels,
        parameter_ranges=eval_ranges, 
        parameter_values=eval_values,
        n=1000,
    )

    print("Génération des graphiques d'évaluation pour le LogSpace Softplus MLP...")
    class ExpWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return torch.exp(self.base_model(x))
            
    wrapped_log_model = ExpWrapper(model_log)
    
    plot_model_evaluation(
        dataset=dataset,
        model=wrapped_log_model,
        parameter_labels=dataset.parameter_labels,
        parameter_ranges=eval_ranges, 
        parameter_values=eval_values,
        n=1000,
    )

if __name__ == "__main__":
    main()
