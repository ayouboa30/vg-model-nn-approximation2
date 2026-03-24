import time

import torch
import numpy as np

from cuda_vg import VGPricingDataset

def mare_fn(y_hat: torch.Tensor, y: torch.Tensor, epsilon=1e-4):
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

    ax = fig.add_subplot(gs[0, 0:10])

    axes_delta = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
        fig.add_subplot(gs[1, 6:8]),
        fig.add_subplot(gs[1, 8:10]),
    ]

    x_queue = []
    y_queue = []
    ic_queue = []

    time_cuda_vg_sampling = dataset.time_vg_sampling

    for _ in range(n):
        x, y, ic = next(dataset)
        x_queue.append(x)
        y_queue.append(y)
        ic_queue.append(ic)

    time_cuda_vg_sampling = dataset.time_vg_sampling - time_cuda_vg_sampling

    x = torch.stack(x_queue)
    y = torch.stack(y_queue)
    ic = torch.stack(ic_queue)

    time_model_vg_sampling = time.time()

    model.eval()
    with torch.no_grad():
        y_hat = model(x.to(next(model.parameters()).device)).to(x.device)

    time_model_vg_sampling = time.time() - time_model_vg_sampling

    print(f"MC sampling time    : {time_cuda_vg_sampling/n:.8f}s/sample")
    print(f"Model sampling time : {time_model_vg_sampling/n:.8f}s/sample")

    rmse = torch.mean((y_hat - y) ** 2, dim=0)
    mare = mare_fn(y_hat, y, epsilon=0.001)

    ax.errorbar(
        y.cpu().numpy().flatten(), 
        y_hat.cpu().numpy().flatten(),  
        yerr=1.98 * ic.cpu().numpy().flatten(), 
        fmt="o",
        markerfacecolor="none", 
        color="black",
        ecolor="black", 
        elinewidth=0.33, 
        label="MC CI",
        alpha=0.5, 
    )
    ax.plot(
        torch.stack([torch.min(y), torch.max(y)]).cpu().numpy(),
        torch.stack([torch.min(y_hat), torch.max(y_hat)]).cpu().numpy(),
        "red", 
        alpha=0.5
    )
    
    ax.set_title(f"Price\nMARE={mare[0].item():.5f} | MSE={rmse[0].item():.5f}")
    ax.set_xlabel(f"MC")
    ax.set_ylabel(f"Model")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

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
    plt.savefig("model_evaluation.png")
    plt.show()
