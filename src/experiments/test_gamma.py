from typing import (
    Callable,
    List
)

import numpy as np
import warnings
from scipy import stats

def kstest_gamma(
    gamma: Callable[[np.typing.ArrayLike], np.typing.ArrayLike], 
    a: List[float] = [0.001, 0.01, 0.1, 0.5, 0.99, 0.999, 1, 1.001, 1.01, 1.1, 2., 10., 100., 1000.], 
    alpha: float = 0.05
):
    for shape in a:
        _, p_value = stats.kstest(gamma(1000), "gamma", args=(shape,))

        if p_value <= alpha:
            warnings.warn(f"Test rejects L = Gamma(a={shape}) with p={p_value}")

def kstest_gamma_min_safe_shape(
    gamma: Callable[[np.typing.ArrayLike], np.typing.ArrayLike], 
    epsilon: float = np.finfo(np.float32).eps.item(), 
    alpha: float = 0.05
):
    a_pass = 0.1
    a_fail = 0.

    p_value_pass = 0.

    while a_pass - a_fail > epsilon:
        a = (a_pass + a_fail) / 2

        x = gamma(1000)
        _, p_value = stats.kstest(x, "gamma", args=(a,))

        if p_value <= alpha:
            a_fail = a
        else:
            a_pass = a
            p_value_pass = p_value

    print(f"Test-based minimum reliable Gamma shape found is a={a_pass} with p={p_value_pass} and epsilon={epsilon}")
            
    return a_pass
