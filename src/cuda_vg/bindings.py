import ctypes
import torch
import warnings
import math

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

def cuda_vg_pricing(
    T: float,
    K: float,
    sigma: float,
    theta: float,
    kappa: float,
    mc_steps: int,
    random_state: CudaRNG
):
    if not hasattr(random_state.lib.cuda_vg_pricing, "argtypes"):
        random_state.lib.cuda_vg_pricing.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.POINTER(CudaRNGStructure), 
        ]
        random_state.lib.cuda_vg_pricing.restype = None

    if random_state.n < mc_steps:
        raise ValueError("Not enough memory allocated to CudaRNG")
    
    if (T / kappa) < 0.002572:
        warnings.warn(
            f"Ratio T / \\kappa ={T / kappa} is below safe threshold for Johnk's Gamma sampling method.",
            RuntimeWarning
        )
    

    x_mc = torch.empty(mc_steps, device="cuda", dtype=torch.float32)
    x_mc_ptr = ctypes.cast(x_mc.data_ptr(), ctypes.POINTER(ctypes.c_float))

    random_state.lib.cuda_vg_pricing(
        x_mc_ptr,
        ctypes.c_float(T),
        ctypes.c_float(K),
        ctypes.c_float(sigma),
        ctypes.c_float(theta),
        ctypes.c_float(kappa),
        ctypes.c_int(mc_steps),
        random_state.handle
    )

    return torch.mean(x_mc).item(), (torch.std(x_mc) / math.sqrt(len(x_mc))).item()
