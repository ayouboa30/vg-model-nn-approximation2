import time

import torch
import numpy as np

from cuda_vg import VGPricingDataset

def mare_fn(y_hat: torch.Tensor, y: torch.Tensor, epsilon=1e-3):
    diff = torch.abs(y_hat - y)
    ares = torch.where(diff < epsilon, 0., diff / torch.clamp(y, min=epsilon))
    return torch.mean(ares, dim=0)

def plot_model_evaluation(
    dataset: VGPricingDataset,
    model: torch.nn.Module, 
    parameter_labels: list, 
    parameter_ranges: list, 
    parameter_values: list,
    n: int,
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 10)

    ax1 = fig.add_subplot(gs[0, 0:5])
    ax2 = fig.add_subplot(gs[0, 5:10])

    axes_delta = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
        fig.add_subplot(gs[1, 6:8]),
        fig.add_subplot(gs[1, 8:10]),
    ]

    x_queue = []
    y_queue = []

    time_cuda_vg_sampling = dataset.time_vg_sampling

    for _ in range(n):
        x, y = next(dataset)
        x_queue.append(x)
        y_queue.append(y)

    time_cuda_vg_sampling = dataset.time_vg_sampling - time_cuda_vg_sampling

    x = torch.stack(x_queue)
    y = torch.stack(y_queue)

    time_model_vg_sampling = time.time()

    model.eval()
    with torch.no_grad():
        y_hat = model(x.to(next(model.parameters()).device)).to(x.device)

    time_model_vg_sampling = time.time() - time_model_vg_sampling

    print(f"MC sampling time    : {time_cuda_vg_sampling/n:.8f}s/sample")
    print(f"Model sampling time : {time_model_vg_sampling/n:.8f}s/sample")

    rmse = torch.sqrt(torch.mean((y_hat - y) ** 2, dim=0))
    mare = mare_fn(y_hat, y, epsilon=0.001)

    ax1.errorbar(
        y[:,0], 
        y_hat[:,0], 
        yerr=y[:, 1], 
        fmt="o",
        markerfacecolor="none", 
        color="black",
        ecolor="black", 
        elinewidth=0.33, 
        label="MC CI",
        alpha=0.5, 
    )
    ax1.plot(
        torch.stack([torch.min(y[:,0]), torch.max(y[:,0])]),
        torch.stack([torch.min(y_hat[:,0]), torch.max(y_hat[:,0])]),
        "red", 
        alpha=0.5
    )
    
    ax1.set_title(f"Price\nMA(R)E={mare[0].item():.2f} | (R)MSE={rmse[0].item():.2f}")
    ax1.set_xlabel(f"MC")
    ax1.set_ylabel(f"Model")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend()

    ax2.scatter(
        y[:, 1], 
        y_hat[:, 1], 
        s=36,
        facecolors="none",
        edgecolors="black",
        alpha=0.5, 
    )
    ax2.plot(
        torch.stack([torch.min(y[:,1]), torch.max(y[:,1])]),
        torch.stack([torch.min(y_hat[:,1]), torch.max(y_hat[:,1])]),
        "red", 
        alpha=0.5
    )
    
    ax2.set_title(f"CI\nMA(R)E={mare[1].item():.2f} | (R)MSE={rmse[1].item():.2f}")
    ax2.set_xlabel(f"MC")
    ax2.set_ylabel(f"Model")
    ax2.grid(True, linestyle=":", alpha=0.5)

    base_x = torch.tensor(parameter_values, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0)

    for i in range(len(parameter_labels)):
        parameter_range = np.linspace(parameter_ranges[i][0], parameter_ranges[i][1], n)
        
        input_batch = base_x.repeat(n, 1)
        input_batch[:, i] = torch.tensor(parameter_range, dtype=torch.float32)
        input_batch.requires_grad_(True)

        output = model(input_batch)

        gradients = torch.autograd.grad(
            outputs=output, 
            inputs=input_batch,
            grad_outputs=torch.ones_like(output),
            create_graph=False
        )[0]

        delta_i = gradients[:, i].detach().cpu().numpy()

        axes_delta[i].plot(parameter_range, delta_i, color="black")
        axes_delta[i].set_title(rf"$\frac{{\partial \hat{{y}}}}{{\partial {('\\' + parameter_labels[i]) if len(parameter_labels[i]) > 1 else parameter_labels[i]}}}$")
        axes_delta[i].axvline(parameter_values[i], color="red", alpha=0.5, label="Reference")
        axes_delta[i].legend()

    plt.tight_layout()
    plt.show()