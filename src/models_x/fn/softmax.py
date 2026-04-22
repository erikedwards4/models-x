"""
JAX functional equivalent of torch.nn.Softmax.

Applies the usual Softmax along axis.
Note: slightly faster than built-in jax.nn.softmax, but
use jax.nn.softmax if need the where input for masking!

Note: no performance gain to hard-code axis = -1,
and no difference in the JAXPR.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["softmax"]


def softmax(arr: Float[Array, "..."],                   # noqa: F722
            axis: int = -1,
            ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape (ndim > 0)
    """
    maxs = jnp.max(a=arr, axis=axis, keepdims=True)     # ... x 1
    exps = jnp.exp(arr - maxs)                          # ... x D
    sums = exps.sum(axis=axis, keepdims=True)           # ... x 1
    return exps / sums                                  # ... x D
