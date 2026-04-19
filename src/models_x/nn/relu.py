"""
JAX functional equivalent of torch.nn.ReLU.

Applies the usual ReLU (element-wise rectification).
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["relu"]


def relu(batch: Float[Array, "..."],
         ) -> Float[Array, "..."]:
    """
    batch: JAX Array of any shape
    """
    return jnp.maximum(0, batch)
