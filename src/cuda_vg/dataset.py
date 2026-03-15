from typing import (
    Callable,
    Union,
    Optional
)

import os
from collections import deque
import time

import torch
import numpy as np

from .bindings import (
    CudaRNG,
    cuda_vg_pricing
)

class VGPricingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        T: Union[float, Callable[[int], np.typing.NDArray]],
        K: Union[float, Callable[[int], np.typing.NDArray]],
        sigma: Union[float, Callable[[int], np.typing.NDArray]],
        theta: Union[float, Callable[[int], np.typing.NDArray]],
        kappa: Union[float, Callable[[int], np.typing.NDArray]],
        mc_steps: int = 100_000,
        lib_file: str = os.path.join(os.path.dirname(__file__), "vg.so"),
        queue_length: int = 1000
    ):
        super().__init__()

        def make_prior(prior: Union[float, Callable[[int], np.typing.NDArray]]) -> Callable[[int], np.typing.NDArray]:
            if callable(prior):
                return prior
            else:
                return lambda shape: np.full(shape, prior)

        self.T = make_prior(T)
        self.K = make_prior(K)
        self.sigma = make_prior(sigma)
        self.theta = make_prior(theta)
        self.kappa = make_prior(kappa)
        self.mc_steps = mc_steps

        self.random_state = CudaRNG(lib_file, torch.initial_seed(), mc_steps)

        self.time_prior_sampling = 0.
        self.time_vg_sampling = 0.
        self.samples = 0

        self.queue_length = queue_length

        self.params_queue = torch.empty((self.queue_length, 5), dtype=torch.float32)
        self.params_queue_idx = self.queue_length

    @property
    def parameter_labels(self):
        return ["T", "K", "sigma", "theta", "kappa"]

    def __iter__(self):
        return self
    
    def __next__(self):
        self.samples += 1

        if self.params_queue_idx < 0 or self.params_queue_idx >= len(self.params_queue):
            time_prior_sampling = time.time()

            np.copyto(
                self.params_queue.numpy(),
                np.stack([
                    self.T(self.queue_length), 
                    self.K(self.queue_length), 
                    self.sigma(self.queue_length), 
                    self.theta(self.queue_length), 
                    self.kappa(self.queue_length)
                ], axis=1).astype(np.float32)
            )

            self.params_queue_idx = 0

            self.time_prior_sampling += time.time() - time_prior_sampling

        x = self.params_queue[self.params_queue_idx]
        self.params_queue_idx += 1

        time_vg_sampling = time.time()

        y, y_ic = cuda_vg_pricing(
            T=x[0].item(), 
            K=x[1].item(), 
            sigma=x[2].item(), 
            theta=x[3].item(), 
            kappa=x[4].item(), 
            mc_steps=self.mc_steps, 
            random_state=self.random_state
        )

        self.time_vg_sampling += time.time() - time_vg_sampling

        return x, torch.tensor([y, y_ic])
