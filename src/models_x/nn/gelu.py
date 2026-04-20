"""
JAX functional equivalent of torch.nn.GELU with approximate=False,
which is called "gelu" in HuggingFace.

This is the original GELU using the Gaussian error function (erf),
not the "new_gelu" using the tanh approximation (see gelu_new.py).

Note: the JAX compiler (XLA) does constant folding,
so isqrt2 = -np.sqrt(0.5) gets computed only once,
Thus, there is no performance gain to making a class,
and storing a self.isqrt2 = ... Also, it is idiomatic
JAX to use functions when no params dict.

See gelu_new.py for reason to use numpy for constants.
"""

import numpy as np
from jax.scipy.special import erf
from jaxtyping import Float, Array

__all__ = ["gelu"]


def gelu(arr: Float[Array, "..."],
         ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape
    """
    isqrt2 = -np.sqrt(0.5).astype(arr.dtype)
    return 0.5 * arr * (1.0 + erf(isqrt2 * arr))  # ...
