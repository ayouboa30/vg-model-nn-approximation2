from typing import (
    Union
)

import numpy as np
import torch

def plot_process(
    t: Union[torch.Tensor, np.ndarray],
    x: Union[torch.Tensor, np.ndarray],
):
    import matplotlib.pyplot as plt

    if isinstance(t, torch.Tensor): t = t.cpu().numpy()
    if isinstance(x, torch.Tensor): x = x.cpu().numpy()

    ax = plt.subplot()

    ax.plot(
        t, 
        x, 
        label="VG Process", 
        marker=".", 
        markerfacecolor="none",
        linestyle="none", 
        color="black"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    import torch

    from src.cuda_vg import cuda_vg_process, CudaRNG

    n = 1024
    t = 10

    lib_file = os.path.join(os.path.dirname(__file__), "..", "cuda_vg", "vg.so")
    random_state = CudaRNG(lib_file, torch.initial_seed(), n)

    x = cuda_vg_process(n, dt=t / n, sigma=0.2, theta=-0.1, kappa=1., random_state=random_state)

    plot_process((torch.ones(n,) * t / n).cumsum_(0), x)
