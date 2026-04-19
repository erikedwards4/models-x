"""
JAX functional equivalent of torch.nn.Dropout.

Applies usual (element-wise) Dropout if training.
Masks and scales to maintain the same expected value.
"""

from jax.random import bernoulli
from jaxtyping import Float, Array

__all__ = ["dropout"]


def dropout(batch: Float[Array, "..."],
            *,
            p: float = 0.0,
            key: Array | None = None,
            deterministic: bool = True,
            ) -> Float[Array, "..."]:
    """
    p: dropout probability in [0.0, 1.0)
       0.0 --> no dropout
    key: JAX PRNG key
         None --> no dropout
    deterministic: similar to eval vs. train mode
                   True  --> no dropout (eval mode)
                   False --> use dropout (train mode)
    """
    # Only if training and valid
    if not deterministic and isinstance(key, Array) and 0.0 < p < 0.999999:
        # Keep prob (p is p_drop)
        p_keep = 1.0 - p

        # Bernoulli mask
        mask = bernoulli(key=key,
                         p=p_keep,
                         shape=batch.shape)

        # Apply mask and scale
        batch = (batch * mask) / p_keep

    return batch
