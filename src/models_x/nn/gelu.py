"""
JAX functional equivalent of torch.nn.GELU.

Applies the usual GELU (element-wise rectification).

Note: the JAX compiler (XLA) does constant folding,
so pi_2 = jnp.sqrt(2/jnp.pi) get computed only once,
and the final number (0.7978...) is baked in. Thus,
there is no performance gain to making a class GELU,
and storing a self.pi_2 = ...
"""

from jax.scipy.special import erf
import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["gelu"]


def gelu(batch: Float[Array, "..."],
         approximate: bool = True,
         ) -> Float[Array, "..."]:
    """
    batch: JAX Array of any shape
    approximate: if to use the tanh approximation
                 True  --> use tanh
                 False --> use erf
    """
    if approximate:
        pi_2 = jnp.sqrt(2 / jnp.pi)
        return 0.5 * batch * \
            (1 + jnp.tanh(pi_2 * (batch + 0.044715*jnp.power(batch, 3))))

    isqrt_2 = -1.0 / jnp.sqrt(2)
    return 0.5 * batch * (1 + erf(batch * isqrt_2))
