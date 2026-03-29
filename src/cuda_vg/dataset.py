from typing import (
    Callable,
    Union,
    Optional,
    List
)

import os
from collections import deque
import time

import torch
import numpy as np

from src.cuda_vg.bindings import (
    CudaRNG,
    cuda_batched_vg_pricing
)

class VGPricingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        T: Union[float, Callable[[int], torch.Tensor]],
        K: Union[float, Callable[[int], torch.Tensor]],
        sigma: Union[float, Callable[[int], torch.Tensor]],
        theta: Union[float, Callable[[int], torch.Tensor]],
        kappa: Union[float, Callable[[int], torch.Tensor]],
        mc_steps: int = 32_768,
        lib_file: str = os.path.join(os.path.dirname(__file__), "vg.so"),
        queue_size: int = 256
    ):
        super().__init__()

        def make_prior(prior: Union[float, Callable[[int], torch.Tensor]]) -> Callable[[int], torch.Tensor]:
            if callable(prior):
                return prior
            else:
                return lambda shape: torch.full(shape, prior, device="cuda")

        self.T = make_prior(T)
        self.K = make_prior(K)
        self.sigma = make_prior(sigma)
        self.theta = make_prior(theta)
        self.kappa = make_prior(kappa)
        self.mc_steps = mc_steps

        self.random_state = CudaRNG(lib_file, torch.initial_seed(), mc_steps * queue_size)

        self.time_prior_sampling = 0.
        self.time_vg_sampling = 0.
        self.samples = 0

        self.queue_size = queue_size
        self.queue_idx = self.queue_size

        self.samples_buffer = torch.empty((self.queue_size, mc_steps), dtype=torch.float32, device="cuda")

        self.params_queues: List[torch.Tensor]
        self.samples_queue: List[torch.Tensor]

    @property
    def parameter_labels(self):
        return ["T", "K", "sigma", "theta", "kappa"]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.queue_idx < 0 or self.queue_idx >= self.queue_size:

            self.samples += self.queue_size
            time_prior_sampling = time.time()

            self.params_queues = [
                self.T(self.queue_size), 
                self.K(self.queue_size), 
                self.sigma(self.queue_size), 
                self.theta(self.queue_size), 
                self.kappa(self.queue_size)
            ]

            self.time_prior_sampling += time.time() - time_prior_sampling

            time_vg_sampling = time.time()

            y, ic = cuda_batched_vg_pricing(
                T=self.params_queues[0], 
                K=self.params_queues[1], 
                sigma=self.params_queues[2], 
                theta=self.params_queues[3], 
                kappa=self.params_queues[4], 
                mc_steps=self.mc_steps,
                random_state=self.random_state,
                buffer=self.samples_buffer
            )

            self.samples_queue = [y, ic]

            self.time_vg_sampling += time.time() - time_vg_sampling

            self.queue_idx = 0

        x = torch.tensor([param_queue[self.queue_idx] for param_queue in self.params_queues], device="cuda")
        y = self.samples_queue[0][self.queue_idx]
        ic = self.samples_queue[1][self.queue_idx]

        self.queue_idx += 1

        return x, y.unsqueeze_(0), ic.unsqueeze_(0)


if __name__ == "__main__":
    import os
    import csv

    device = "cuda"
    mc_steps = 32_768

    param_priors = {
        "T": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0),
        "K": lambda size: torch.full((size,), 1., device=device), # np.random.normal(loc=1, scale=0.001, size=size)
        "sigma": lambda size: torch.empty((size,), device=device).uniform_(0.05, 0.6),
        "theta": lambda size: torch.normal(mean=-0.1, std=1.0, size=(size,), device=device).clamp_(-0.5, 0.2),
        "kappa": lambda size: torch.empty((size,), device=device).uniform_(0.1, 2.0).exp_(),
    }


    dataset = VGPricingDataset(**param_priors, mc_steps=mc_steps)

    os.makedirs(".cache", exist_ok=True)
    with open(os.path.join(".cache", "dataset.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["T", "K", "sigma", "theta", "kappa", "price", "ic"])

            for _ in range(1000):
                x, y, ic = next(dataset)
                writer.writerow(torch.cat([x, y, ic]).tolist())