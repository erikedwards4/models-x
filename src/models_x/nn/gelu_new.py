"""
JAX functional equivalent of torch.nn.GELU with approximate=True,
which is called "gelu_new" or "new_gelu" in HuggingFace.

See gelu.py for the original GELU with approximate=False.

Note: the JAX compiler (XLA) does constant folding,
so pi_2 = np.sqrt(2/np.pi) gets computed only once,
and the final number (0.7978...) is baked in. Thus,
there is no performance gain to making a class, and
storing a self.pi_2 = ...

However, using np to get pi_2 is faster, and this is
also how jnp source code computes constants. It can
also be seen to give shorter intermediate representation
with jax.make_jaxpr (JAX programming representation).
"""

import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Array

__all__ = ["gelu_new"]


def gelu_new(arr: Float[Array, "..."],
             ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape
    """
    # pi_2 = jnp.sqrt(2.0/jnp.pi).astype(arr.dtype)
    pi_2 = np.sqrt(2.0/np.pi).astype(arr.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(pi_2 * (arr + 0.044715*(arr**3))))  # ...
    return arr * cdf                                                # ...
