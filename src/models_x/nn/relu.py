"""
JAX functional equivalent of torch.nn.ReLU.

Applies the usual ReLU (element-wise rectification).
Note: slightly faster than built-in jax.nn.relu, but
use jax.nn.relu if concerned about the derivative at 0.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["relu"]


def relu(arr: Float[Array, "..."],
         ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape
    """
    return jnp.maximum(0.0, arr)        # ...
