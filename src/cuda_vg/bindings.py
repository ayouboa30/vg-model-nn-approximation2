from typing import (
    Optional
)

import ctypes
import torch
import warnings
import math
import os

class CudaRNGStructure(ctypes.Structure):
    _fields_ = [
        ("states", ctypes.c_void_p),
        ("n", ctypes.c_int)
    ]

class CudaRNG:
    def __init__(self, lib_path: str, seed: int, n: int):
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.cuda_init_rng.argtypes = [ctypes.c_ulong, ctypes.c_int]
        self.lib.cuda_init_rng.restype = ctypes.POINTER(CudaRNGStructure)

        self.lib.cuda_cleanup_rng.argtypes = [ctypes.POINTER(CudaRNGStructure)]
        
        self.handle = self.lib.cuda_init_rng(seed, n)
        self.n = n

    def __del__(self):
        if hasattr(self, "handle"):
            self.lib.cuda_cleanup_rng(self.handle)
            del self.handle


def safe_tensor(t: torch.Tensor, raise_error: bool = False) -> torch.Tensor:
    if not t.is_cuda:
        message = "Tensor is not on device memory"

        if raise_error: raise RuntimeError(message)
        else: 
            warnings.warn(message, RuntimeWarning)
            t = t.to("cuda")

    if t.dtype != torch.float32:
        message = "Tensor is not float32"

        if raise_error: raise RuntimeError(message)
        else: 
            warnings.warn(message, RuntimeWarning)
            t = t.to(torch.float32)

    if not t.is_contiguous():
        message = "Tensor is not contiguous"

        if raise_error: raise RuntimeError(message)
        else: 
            warnings.warn(message, RuntimeWarning)
            t = t.contiguous()

    return t

def cuda_gamma(n: int, a: float, random_state: CudaRNG):
    if not hasattr(random_state.lib.cuda_gamma, "argtypes"):
        random_state.lib.cuda_gamma.argtypes = [
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(CudaRNGStructure), 
        ]

    if random_state.n < n:
        raise ValueError("Not enough memory allocated to CudaRNG")
    
    if a < 0.002572:
        warnings.warn(
            f"Gamma shape={a} is below safe threshold for Johnk's method.",
            RuntimeWarning
        )

    x = torch.empty(n, device="cuda", dtype=torch.float32)
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))

    random_state.lib.cuda_gamma(x_ptr, n, ctypes.c_float(a), random_state.handle)

    return x

def cuda_vg_process(
    n: int,
    dt: float,
    sigma: float,
    theta: float,
    kappa: float,
    random_state: CudaRNG,
):
    if not hasattr(random_state.lib.cuda_vg_process, "argtypes"):
        random_state.lib.cuda_vg_process.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.POINTER(CudaRNGStructure), 
        ]
        random_state.lib.cuda_vg_process.restype = None

    x = torch.empty((n,), device="cuda", dtype=torch.float32)

    if random_state.n < n:
        raise ValueError("Not enough memory allocated to CudaRNG")
    
    if dt < 0.002572:
        warnings.warn(
            f"Time steps are below safe threshold (0.002572) for Johnk's Gamma sampling method.",
            RuntimeWarning
        )

    random_state.lib.cuda_vg_process(
        ctypes.cast(safe_tensor(x, raise_error=True).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(dt),
        ctypes.c_float(sigma),
        ctypes.c_float(theta),
        ctypes.c_float(kappa),
        ctypes.c_int(n),
        random_state.handle
    )

    x.cumsum_(0)

    return x

def cuda_batched_vg_pricing(
    T: torch.Tensor,
    K: torch.Tensor,
    sigma: torch.Tensor,
    theta: torch.Tensor,
    kappa: torch.Tensor,
    mc_steps: int,
    random_state: CudaRNG,
    buffer: Optional[torch.Tensor] = None
):
    if not hasattr(random_state.lib.cuda_batched_vg_pricing, "argtypes"):
        random_state.lib.cuda_batched_vg_pricing.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CudaRNGStructure), 
        ]
        random_state.lib.cuda_batched_vg_pricing.restype = None

    batch_size = len(T)

    if random_state.n < (mc_steps * batch_size):
        raise ValueError("Not enough memory allocated to CudaRNG")
    
    if torch.any((T / kappa) < 0.002572):
        warnings.warn(
            f"Ratio T / \\kappa is below safe threshold (0.002572) for Johnk's Gamma sampling method.",
            RuntimeWarning
        )

    if buffer is None:
        buffer = torch.empty((batch_size, mc_steps), device="cuda", dtype=torch.float32)

    random_state.lib.cuda_batched_vg_pricing(
        ctypes.cast(safe_tensor(buffer, raise_error=True).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(safe_tensor(T).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(safe_tensor(K).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(safe_tensor(sigma).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(safe_tensor(theta).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(safe_tensor(kappa).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(batch_size),
        ctypes.c_int(mc_steps),
        random_state.handle
    )

    return torch.cat([
        torch.mean(buffer, dim=1, keepdim=True),
        torch.std(buffer, dim=1, keepdim=True) / math.sqrt(mc_steps),
    ], dim=1)


def test_rng():
    rng = CudaRNG(os.path.join(os.path.dirname(__file__), "vg.so"), 0, 16)

    params = dict(
        T=torch.tensor([1.], dtype=torch.float32, device="cuda"),
        K=torch.tensor([1.], dtype=torch.float32, device="cuda"),
        sigma=torch.tensor([0.2], dtype=torch.float32, device="cuda"),
        theta=torch.tensor([-0.1], dtype=torch.float32, device="cuda"),
        kappa=torch.tensor([1.], dtype=torch.float32, device="cuda"),
    )

    print(cuda_batched_vg_pricing(**params, mc_steps=16, random_state=rng))
    print(cuda_batched_vg_pricing(**params, mc_steps=16, random_state=rng))

if __name__ == "__main__":
    test_rng()