from typing import (
    List,
    Optional
)

def plot_learning_curves(train_losses: List[float], val_losses: List[float], test_loss: float, learning_rates: Optional[List[float]] = None):
    import matplotlib.pyplot as plt

    has_lr = learning_rates and len(learning_rates) > 0
    fig, axes = plt.subplots(has_lr + 1, 1, figsize=(10, 8 if has_lr else 6), sharex=False)
    
    if not has_lr:
        ax_loss = axes
    else:
        ax_loss = axes[0]
        ax_lr = axes[1]

    epochs = range(1, len(train_losses) + 1)

    ax_loss.plot(epochs, train_losses, label="Train Loss", color="black")
    ax_loss.plot(epochs, val_losses, label="Validation Loss", color="darkred")
    ax_loss.axhline(y=test_loss, label=f"Test Loss ({test_loss:.4f})", color="red")
    
    ax_loss.set_title("Learning Curve (Loss)")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    if has_lr:
        lr_epochs = range(1, len(learning_rates) + 1)
        ax_lr.plot(lr_epochs, learning_rates, label="Learning Rate", color="black")
        
        if max(learning_rates) / (min(learning_rates) + 1e-12) > 10:
            ax_lr.set_yscale("log")
            ax_lr.set_ylabel("LR (log scale)")
        else:
            ax_lr.set_ylabel("LR")
            
        ax_lr.set_title("Learning Rate Schedule")
        ax_lr.set_xlabel("Steps")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend()

    plt.tight_layout()
    plt.show()
