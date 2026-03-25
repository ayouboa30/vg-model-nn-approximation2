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
from metrics import ThresholdedWeightedMSE, MonotonyLoss, ConvexityLoss, CombinedLoss
from models import Linear, MLP, PICNN
from experiments import plot_model_evaluation, plot_learning_curves

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loss_fn, loader, device=None, mu=None, sigma=None):
    model.eval()
    x, y, ic = next(iter(loader))
    x, y, ic = x.to(device), y.to(device), ic.to(device)

    x_norm = (x - mu) / (sigma + 1e-6)
    x_norm = x_norm.detach().requires_grad_(True)

    y_hat = model(x_norm)
    loss = loss_fn(x_norm, y_hat, y, ic)
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
    max_epoch = 800
    device = "cuda"

    mc_steps = 32_768
    # param_priors = {
    #     "T": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0),
    #     "K": lambda size: torch.full((size,), 1., device=device), # np.random.normal(loc=1, scale=0.001, size=size)
    #     "sigma": lambda size: torch.empty((size,), device=device).uniform_(0.05, 0.6),
    #     "theta": lambda size: torch.normal(mean=-0.1, std=1.0, size=(size,), device=device).clamp_(-0.5, 0.2),
    #     "kappa": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0).exp_(),
    # }

    param_priors = {
        "T": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0),
        "K": lambda size: torch.empty((size,), device=device).uniform_(0.8, 1.2), 
        "sigma": lambda size: torch.empty((size,), device=device).uniform_(0.05, 0.6),
        "theta": lambda size: torch.empty((size,), device=device).uniform_(-0.5, 0.05), 
        "kappa": lambda size: torch.empty((size,), device=device).uniform_(0.1, 1.0), 
    }

    set_seed(seed)

    dataset = VGPricingDataset(**param_priors, mc_steps=mc_steps)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # ThresholdedWeightedMSE uses our MLE derived weighted MSE,
    # it will easily reach +1e5 in early epochs, so gradient clipping
    # is quite mandatory, see the training loop.
    # It is mathematically sound however, and if minimized long enough,
    # leads to a very good MARE. 
    #
    # TODO : Write a clean .md

    # loss_fn = CombinedLoss([(torch.nn.MSELoss(), 1.)])
    # loss_fn = CombinedLoss([(ThresholdedWeightedMSE(1e-8), 1.)])
    # loss_fn = CombinedLoss([
    #     (torch.nn.MSELoss(), 1.),
    #     (MonotonyLoss(1, increasing=False), 1.),
    #     (MonotonyLoss(0, increasing=True), 1.),
    #     (ConvexityLoss(0, convex=True), 1.),
    # ])
    #loss_fn = CombinedLoss([
     #   (ThresholdedWeightedMSE(precision=1e-8), 1.),
      #  (MonotonyLoss(1, increasing=False), 10.),
       # (MonotonyLoss(0, increasing=True), 10.),
        #(ConvexityLoss(1, convex=True), 0.),
    #])
    # Phase 1 : Uniquement la précision (MSE)
    loss_fn_phase1 = CombinedLoss([
        (torch.nn.MSELoss(), 1.),
        (MonotonyLoss(1, increasing=False), 0.0), # Désactivé
        (MonotonyLoss(0, increasing=True), 0.0),  # Désactivé
        (ConvexityLoss(1, convex=True), 0.),
    ])

    # Phase 2 : Précision + Fortes contraintes physiques
    loss_fn_phase2 = CombinedLoss([
        (ThresholdedWeightedMSE(precision=1e-4), 1.),
        (MonotonyLoss(1, increasing=False), 10.0), # Fortement pénalisé
        (MonotonyLoss(0, increasing=True), 10.0),  # Fortement pénalisé
        (ConvexityLoss(1, convex=True), 0.),       # Géré par le PICNN
    ])

    # On démarre avec la Phase 1
    current_loss_fn = loss_fn_phase1
    is_phase_2 = False

    # model = Linear(bias=False, device=device)
    model = PICNN(hidden_dim=128, depth=5, device=device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Learnable parameters : {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    scheduler = None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
         optimizer,
         max_lr=1e-3, 
         steps_per_epoch=1,
         epochs=max_epoch,
         pct_start=0.3,
         anneal_strategy="cos"
     )

    early_stopping = EarlyStopping(
        patience=80,
        monitor="loss",
        mode="min",
        delta=1e-5,
    )

    epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    mu = torch.tensor([1.05, 1.0, 0.325, -0.225, 0.55], device=device)
    sigma = torch.tensor([0.55, 0.11, 0.16, 0.16, 0.26], device=device)

    while epoch < max_epoch:
       
        epoch += 1
        epoch_train_losses = []

        model.train()
        for batch, (x, y, ic) in enumerate(tqdm(loader, total=epoch_size, desc=f"Epoch {epoch}", postfix={
            "phase": "2" if is_phase_2 else "1",
            "train_loss": train_losses[-1] if train_losses else "?" ,
            "val_loss": val_losses[-1] if val_losses else "?" ,
        }, leave=False)):
            if batch >= epoch_size:
                break

            x_norm = ((x - mu) / (sigma + 1e-6)).detach().requires_grad_(True)
        
            optimizer.zero_grad()
            y_hat = model(x_norm) 
            loss = current_loss_fn(x_norm, y_hat, y, ic)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_train_losses.append(loss.item())

        # Fin d'époque : Calcul des métriques
        avg_train_loss = torch.mean(torch.tensor(epoch_train_losses)).item()
        train_losses.append(avg_train_loss)
        
        # Évaluation sur le set de validation
        current_val_loss = evaluate(model, current_loss_fn, loader, device=device, mu=mu, sigma=sigma)
        val_losses.append(current_val_loss)

        learning_rates.append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

        # --- Logique d'Early Stopping et Transition de Phase ---
        if early_stopping(metrics={"loss": current_val_loss}):
            if not is_phase_2:
                print(f"\n[Phase 1 terminée à l'epoch {epoch}] Convergence MSE atteinte.")
                print("--> Passage en Phase 2 : Activation des contraintes de monotonie.")
                
                # 1. Changement de configuration
                is_phase_2 = True
                current_loss_fn = loss_fn_phase2
                
                # 2. Réinitialisation de l'Early Stopping (Crucial pour éviter Index Error / Stagnation)
                early_stopping = EarlyStopping(
                    patience=50,
                    monitor="loss",
                    mode="min",
                    delta=1e-5,
                )
                
                # 3. Passage en mode "Fine-tuning" : LR plus faible et stable
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4 
                scheduler = None # On arrête le OneCycleLR
                
            else:
                print(f"Early stopping définitif à l'epoch : {epoch}")
                break
            
            
    else:
        print(f"Hit max epoch : {epoch}")

    test_loss = evaluate(model, current_loss_fn, loader, device=device, mu=mu, sigma=sigma)
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
        n=1000,
    )

if __name__ == "__main__":
    main()
