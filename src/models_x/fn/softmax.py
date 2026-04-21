"""
JAX functional equivalent of torch.nn.Softmax.

Applies the usual Softmax along axis.
Note: slightly faster than built-in jax.nn.softmax, but
use jax.nn.softmax if need the where input for masking.

Note: no performance gain to hard-code axis = -1,
and no difference in the JAXPR.

Note: this does handle -Inf vals to output 0 where -Inf.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["softmax"]


def softmax(arr: Float[Array, "..."],
            axis: int = -1,
            ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape (ndim > 0)
    """
    maxs = jnp.max(arr, axis=axis, keepdims=True)   # ... x 1
    exps = jnp.exp(arr - maxs)                      # ... x D
    sums = exps.sum(axis=axis, keepdims=True)       # ... x 1
    return exps / sums                              # ... x D
