"""
JAX functional equivalent of torch.nn.Dropout.

If training, applies the usual Dropout (element-wise
mask), and scales to maintain the same expected value.
"""

from jax.random import bernoulli
from jaxtyping import Float, Array

__all__ = ["dropout"]


def dropout(arr: Float[Array, "..."],
            *,
            p: float = 0.0,
            key: Array | None = None
            ) -> Float[Array, "..."]:
    """
    arr: JAX Array of any shape
    p: dropout probability in [0.0, 1.0)
       0.0 --> no dropout (eval mode)
    key: JAX PRNG key
         None --> no dropout (eval mode)
    """
    # Only if training and valid
    if 0.0 < p < 0.999999 and isinstance(key, Array):
        # Keep prob (p is p_drop)
        p_keep = 1.0 - p

        # Bernoulli mask
        mask = bernoulli(key=key,
                         p=p_keep,
                         shape=arr.shape)       # ...

        # Apply mask and scale
        arr = (arr * mask) / p_keep             # ...

    return arr                                  # ...
